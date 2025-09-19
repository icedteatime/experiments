
import torch

import argparse
import einops
import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms

from the_greatest_logging_ever import print, print_lines, summary


name = "Softfloor"

import inspect
description = lambda metadata: f"""
Softfloor function based on sigmoid.

![softfloor](images/_09_softfloor1.svg)

```python
{inspect.getsource(metadata["PythonModule"].softfloor_make)}```
"""

defaults = {
    "batch_size": 64,
    "epochs": 14,
    "learning_rate": 1e-3,
    "log_interval": 200
}


def softfloor_make(factor=1e-4):
    factor = 1 + factor
    equation = sympy.Eq(1, factor*(2 / (1 + sympy.exp(-sympy.symbols("x"))) - 1))
    adjustment = float(sympy.solve(equation)[0])

    def f(xs):
        floor = torch.floor(xs + 0.5)
        xs = xs - floor
        ys = floor + factor*((1 / (1 + torch.exp(-(2*xs)*adjustment))) - 1)

        return ys + 0.5*factor - 0.5

    return f

def wavy_identity_make(factor=0.01):
    assert factor >= 1e-2, f"Value of {factor} violates squiggle-identity aesthetics."

    sf = softfloor_make(factor)

    def f(xs):
        return sf(xs) + 0.5

    return f


class LambdaModule(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=(3, 3)),
            LambdaModule(wavy_identity_make()),
            nn.LayerNorm(normalized_shape=(26, 26)),
            nn.Dropout(0.3),
            nn.AvgPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),

            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3)),
            LambdaModule(softfloor_make()),
            nn.LayerNorm(normalized_shape=(11, 11)),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.AvgPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),

            nn.Flatten(),

            nn.Linear(in_features=400, out_features=128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=128, out_features=10)
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

    transform=transforms.Compose([
        transforms.ToTensor(),
        ])

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