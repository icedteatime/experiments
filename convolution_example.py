
import torch

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from the_greatest_logging_ever import print


name = "Convolution Example"
description = """
Basic convolution.
Example from https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

defaults = {
    "batch_size": 64,
    "epochs": 14,
    "learning_rate": 1e-3,
    "log_interval": 100
}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
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
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
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
                    # "pin_memory": True,
                    "shuffle": True}

    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    dataset1 = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("datasets", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Model().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step(metrics=loss)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example", allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"], help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="number of epochs to train (default: 14)")
    parser.add_argument('--learning-rate', type=float, default=defaults["learning_rate"], help='learning rate (default: 1.0)')
    parser.add_argument('--log-interval', type=int, default=defaults["log_interval"], help='how many batches to wait before logging training status')

    args = parser.parse_args()

    run(**args.__dict__)