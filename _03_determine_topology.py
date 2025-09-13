
import torch

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from the_greatest_logging_ever import print


name = "Determine topology"
description = """
Infer topology of training data. For MNIST that would be close to a 2D grid.
A single linear layer acts as a correlation matrix.

#### Correlations according to weight matrix
![correlation plot](images/_03_determine_topology1.png)

The blue diagonal along the middle means each pixel is strongly correlated with itself. The two red diagonals mean that pixels at a vertical distance of 2 are anticorrelated.

#### Topology based on a threshold
![graph](images/_03_determine_topology2.png)

Since this is the "data topology", it doesn't actually match the grid and isn't planar.

"""

defaults = {
    "batch_size": 2048,
    "epochs": 24,
    "learning_rate": 1e-3,
    "log_interval": 10
}

def permutation_make(shape):
    """
    Create a positional permutation/unpermutation function pair for arrays with a given shape.
    """

    n = np.array(shape).prod().item()

    permutation = dict(zip(range(n), np.random.choice(range(n), n, replace=False).tolist()))
    permutation_inverse = dict(sorted(dict(map(reversed, permutation.items())).items()))
    assert all(permutation_inverse[permutation[i]] == i
               for i in range(n))

    permute = lambda array: array.flatten()[list(permutation.values())].reshape(shape)
    unpermute = lambda array: array.flatten()[list(permutation_inverse.values())].reshape(shape)

    return permute, unpermute

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.linear1 = nn.Linear(in_features=28*28,
                                 out_features=28*28,
                                 bias=False)

    def forward(self, x):
        x = self.linear1(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.mse_loss(output, data)
        loss = ((output - data)**2).sum()**0.5
        loss.backward()
        optimizer.step()
        if batch_index % args.log_interval == 0:
            print({"Epoch": epoch,
                   "Batch": batch_index,
                   "Loss": loss.item()})
    
    return loss.item()


def test(model, device, test_loader, epoch):
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

    print({"Epoch": epoch,
           "TestSetAverageLoss": test_loss / len(test_loader.dataset)})


def run(**kwargs):
    args = argparse.Namespace(**(defaults | kwargs))
    print({"Arguments": args.__dict__})

    device = torch.accelerator.current_accelerator()

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.batch_size}

    accel_kwargs = {# "num_workers": 1,
                    # "persistent_workers": True,
                    "shuffle": True}

    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)

    permute, unpermute = permutation_make(shape=(1, 28, 28))

    transform=transforms.Compose([
        transforms.ToTensor(),
        # lambda x: 2*x - 1,
        # lambda x: x.transpose(-2, -1),
        permute,
        torch.flatten
        ])

    print({"Transform": transform.transforms})

    dataset1 = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("datasets", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    permutation = permute(np.arange(28*28)).flatten()
    unpermutation = unpermute(np.arange(28*28)).flatten()
    print({"Permutation": permutation})

    model = Model().to(device)
    model.permutation = permutation
    model.unpermutation = unpermutation

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step(metrics=loss)

    return model

if __name__ == "__main__":
    run(**defaults)