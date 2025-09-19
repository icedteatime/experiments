
import torch

import argparse
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from typing import Union

from the_greatest_logging_ever import print, print_lines, summary


name = "Staged networks"
description = lambda metadata: f"""
First fully train one network on a task. Freeze its weights and attach a new network.
Train the combined network and repeat.

```
{metadata["ModelDescriptionEllipsized"]}
```
"""

defaults = {
    "batch_size": 64,
    "epochs": 14,
    "learning_rate": 1e-3,
    "log_interval": 200
}


class Expansion(nn.Module):
    @staticmethod
    def forward(module, x, expansion_step):
        if isinstance(module, Expansion):
            return module(x, expansion_step)
        else:
            return module(x)

class ExpansionParallel(Expansion):
    def __init__(self, networks, should_freeze_previous=True):
        super().__init__()

        self.networks = nn.ModuleList(networks)
        self.should_freeze_previous = should_freeze_previous
    
    def forward(self, x, expansion_step):
        active_networks = self.networks[:expansion_step+1]
        if self.should_freeze_previous and len(active_networks) > 1:
            for network in active_networks[:-1]:
                for parameters in network.parameters():
                    parameters.requires_grad = False

        output = []
        for index, network in enumerate(active_networks):
            output.append(Expansion.forward(network, x, expansion_step - index))

        x = torch.cat(output, dim=-1)

        return x

class ExpansionReplace(Expansion):
    def __init__(self, networks):
        super().__init__()

        self.networks = nn.ModuleList(networks)
        
    def forward(self, x, expansion_step):
        active_network = self.networks[expansion_step]

        return Expansion.forward(active_network, x, expansion_step)

class ExpansionSerial(Expansion):
    def __init__(self, networks):
        super().__init__()

        self.networks = nn.ModuleList(networks)
        
    def forward(self, x, expansion_step):
        for node in self.networks:
            x = Expansion.forward(node, x, expansion_step=expansion_step)

        return x

class RandomSample(nn.Module):
    def __init__(self, input_size, subset_size):
        super().__init__()

        self.subset_indices = torch.randperm(input_size)[:subset_size]

    def forward(self, x):
        return x[:, self.subset_indices]

class Residual(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return x + self.network(x)


class Model(Expansion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        Layers = lambda num_layers: nn.Sequential(
            RandomSample(input_size=28*28,
                        subset_size=16),
            *[Residual(
                nn.Sequential(nn.Linear(16, 16),
                            nn.LayerNorm(normalized_shape=16),
                            nn.ReLU()))
            for _ in range(num_layers)],
        )

        # [](block1)
        parallel = ExpansionParallel([Layers(1) for _ in range(10)] +
                                     [Layers(2) for _ in range(5)] +
                                     [Layers(4) for _ in range(5)] +
                                     [Layers(8) for _ in range(5)] +
                                     [Layers(16) for _ in range(5)])

        self.n = len(parallel.networks)

        self.sequence = ExpansionSerial(networks=[
            nn.Flatten(),
            parallel,
            ExpansionReplace([nn.Linear((i+1)*16, 10)
                              for i in range(self.n)])
        ])

    def forward(self, x, expansion_step):
        return Expansion.forward(self.sequence, x, expansion_step)

def train_outer(args, device, train_loader, test_loader):
    model = Model()

    model.to(device)

    max_expansion_steps = model.n
    for expansion_step in range(max_expansion_steps):
        epochs = args.epochs

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)

        for epoch in range(1, epochs + 1):
            print({"LearningRateLog10": np.log10(optimizer.param_groups[0]["lr"])})
            loss = train(args, model, expansion_step, device, train_loader, optimizer, epoch)
            test(model, expansion_step, device, test_loader, epoch)
            scheduler.step(metrics=loss)

    return model


def train(args, model, expansion_step, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, expansion_step)

        loss = F.cross_entropy(output, target)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

        if batch_index % args.log_interval == 0:
            print({"ExpansionStep": expansion_step,
                   "Epoch": epoch,
                   "Batch": batch_index,
                   "Loss": loss.item()})
    
    print({"LossAverage": sum(losses) / len(losses)})
    return sum(losses) / len(losses)


def test(model, expansion_step, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_length = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, expansion_step)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print({"Epoch": epoch,
           "Accuracy": f"{correct/total_length}: {correct}/{total_length}",
           "TestSetAverageLoss": test_loss / total_length})


def run(**kwargs):
    args = argparse.Namespace(**(defaults | kwargs))
    print({"Arguments": args.__dict__})

    device = torch.accelerator.current_accelerator()

    print({"Device": device})

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.batch_size}

    accel_kwargs = {"shuffle": True}

    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)

    # transform = lambda x: torch.tensor(np.array(x, dtype=np.long))
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST("datasets", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST("datasets", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = train_outer(args=args,
                        device=device,
                        train_loader=train_loader,
                        test_loader=test_loader)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-attention", allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"], help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="number of epochs to train (default: 14)")
    parser.add_argument('--learning-rate', type=float, default=defaults["learning_rate"], help='learning rate (default: 1.0)')
    parser.add_argument('--log-interval', type=int, default=defaults["log_interval"], help='how many batches to wait before logging training status')

    args = parser.parse_args()

    run(**(defaults | args.__dict__))