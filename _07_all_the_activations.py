
import torch

import argparse
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms

from the_greatest_logging_ever import print, print_lines, summary


name = "All the activations"
description = """
If you're having trouble deciding between activations, just use all of them. Easy.
Includes a skip connection, which can be considered the null activation, very cool.
"""

defaults = {
    "batch_size": 64,
    "epochs": 14,
    "learning_rate": 1e-3,
    "log_interval": 200
}


class Parallel(nn.Module):
    def __init__(self, networks: list[nn.Module],
                 reduction=lambda x: torch.cat(x, dim=-1)):
        super().__init__()
        self.networks = nn.ModuleList(networks)

        self.reduction = reduction

    def forward(self, x):
        x = self.reduction([network(x)
                            for network in self.networks])
        return x

class Residual(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return x + self.network(x)

class LambdaModule(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

    def __repr__(self):
        import inspect
        source = inspect.getsource(self.function)
        return source.strip()

class Log(nn.Module):
    def __init__(self, label=None):
        super().__init__()
        self.label = label

    def forward(self, x):
        summary(x, label=self.label)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        def Conv1d(k, permute_size=None):
            if permute_size:
                permutation = torch.randperm(permute_size)
                permute_layer = [LambdaModule(lambda x: x[:, permutation])]
            else:
                permute_layer = []

            return nn.Sequential(
                *permute_layer,
                LambdaModule(lambda x: x.unsqueeze(1)),
                nn.Conv1d(in_channels=1, out_channels=k, kernel_size=k, stride=k),
                nn.Flatten(),
                nn.GELU()
            )

        def SoftmaxGrouped(group_size, permute_size=None):
            permute_layer = []
            if permute_size:
                permutation = torch.randperm(permute_size)
                permute_layer = [LambdaModule(lambda x: x[:, permutation])]

            return nn.Sequential(
                *permute_layer,
                LambdaModule(lambda x: F.softmax(x.reshape(-1, x.shape[1]//group_size, group_size), dim=-1).reshape(x.shape)),
                nn.Flatten(),
            )

        activation = lambda input_size: nn.Sequential(
            Parallel([
                LambdaModule(lambda x: x),
                nn.ReLU(),
                nn.GELU(),
                nn.Sigmoid(),
                nn.LeakyReLU(),
                nn.LogSigmoid(),
                SoftmaxGrouped(group_size=4),
                SoftmaxGrouped(group_size=8, permute_size=input_size),
                Conv1d(4),
                Conv1d(8, permute_size=input_size),
            ])
        )

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=2),
            nn.LayerNorm(normalized_shape=(30, 30)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),

            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3)),
            nn.LayerNorm(normalized_shape=(13, 13)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),

            nn.Flatten(),

            nn.Linear(in_features=16*6*6, out_features=128),
            activation(input_size=128),
            nn.Dropout(0.5),
            nn.Linear(in_features=1280, out_features=128),
            activation(input_size=128),
            nn.Dropout(0.5),
            nn.Linear(in_features=1280, out_features=10)
        )

    def forward(self, x):
        x = self.sequence(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

        if batch_index % args.log_interval == 0:
            print({"Epoch": epoch,
                   "Batch": batch_index,
                   "Loss": loss.item()})
    
    print({"LossAverage": sum(losses) / len(losses)})
    return sum(losses) / len(losses)


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_length = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
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

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

    for epoch in range(1, args.epochs + 1):
        print({"LearningRateLog10": np.log10(optimizer.param_groups[0]["lr"])})
        loss = train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step(metrics=loss)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-attention", allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"], help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="number of epochs to train (default: 14)")
    parser.add_argument('--learning-rate', type=float, default=defaults["learning_rate"], help='learning rate (default: 1.0)')
    parser.add_argument('--log-interval', type=int, default=defaults["log_interval"], help='how many batches to wait before logging training status')

    args = parser.parse_args()

    run(**(defaults | args.__dict__))