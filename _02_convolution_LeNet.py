
import torch

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from the_greatest_logging_ever import print, print_lines


name = "LeNet"
description = """
[LeNet](https://en.wikipedia.org/wiki/LeNet)-like.
"""

defaults = {
    "batch_size": 64,
    "epochs": 14,
    "learning_rate": 1e-3,
    "log_interval": 100
}

class LambdaModule(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        scaled_tanh = LambdaModule(function=lambda x: 1.7159*torch.tanh(2*x/3))

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=(5, 5),
                      padding=2),
            scaled_tanh,
            nn.AvgPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),

            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=(5, 5)),
            scaled_tanh,
            nn.AvgPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),

            nn.Flatten(),

            nn.Linear(in_features=400, out_features=120),
            scaled_tanh,
            nn.Linear(in_features=120, out_features=84),
            scaled_tanh,
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_index % args.log_interval == 0:
            print({"Epoch": epoch, "Batch": batch_index, "Loss": loss.item()})
    
    return loss.item()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print({"TestSetAverageLoss": test_loss / len(test_loader.dataset),
           "Accuracy": f"{correct}/{len(test_loader.dataset)}",
           "AccuracyPercent": 100 * correct / len(test_loader.dataset)})


def run(**kwargs):
    args = argparse.Namespace(**(defaults | kwargs))
    print({"Arguments": args.__dict__})

    device = torch.accelerator.current_accelerator()

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.batch_size}

    accel_kwargs = {"num_workers": 1,
                    "persistent_workers": True,
                    "shuffle": True}

    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    dataset1 = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("datasets", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step(metrics=loss)

    return model

if __name__ == "__main__":
    run(**defaults)