
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


name = "Autoencoder"
description = lambda metadata: f"""
Basic autoencoder.

#### t-SNE plot of encoded digits
![tSNE](images/_06_autoencoder1.png)

```
{metadata["ModelDescription"]}
```
"""

defaults = {
    "batch_size": 64,
    "epochs": 14,
    "learning_rate": 1e-3,
    "log_interval": 200
}


def structural_similarity(image1, image2):
    """
    https://en.wikipedia.org/wiki/Structural_similarity_index_measure 
    """

    batch, channel, height, width = image1.shape
    image_size = channel*height*width

    image1 = image1.flatten(start_dim=1)
    image2 = image2.flatten(start_dim=1)

    ux = image1.mean(dim=-1)
    uy = image2.mean(dim=-1)
    cov = ((image1 - ux.unsqueeze(1)) * (image2 - uy.unsqueeze(1))).sum(dim=-1) / (image_size - 1)

    L = 1
    c1 = (0.01*L)**2
    c2 = (0.03*L)**2

    numerator = (2*ux*uy + c1) * (2*cov + c2)
    denominator = (ux**2 + uy**2 + c1) * (image1.var(dim=-1) + image2.var(dim=-1) + c2)

    return numerator / denominator


class Parallel(nn.Module):
    def __init__(self, networks,
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

        activation = nn.ReLU

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=4,
                      kernel_size=(2, 2),
                      stride=(2, 2)),
            activation(),
            nn.Conv2d(in_channels=4,
                      out_channels=8,
                      kernel_size=(2, 2),
                      stride=(2, 2)),
            activation(),

            nn.Flatten(),

            nn.Linear(in_features=392, out_features=32),
        )

        self.decode = nn.Sequential(
            nn.Linear(in_features=32, out_features=392),
            activation(),
            nn.Linear(in_features=392, out_features=784),
            LambdaModule(lambda x: x.reshape(-1, 1, 28, 28))
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, data)
        # loss = -structural_similarity(output, data).abs().mean()
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
    # correct = 0
    total_length = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            test_loss += F.mse_loss(output, data, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    print({"Epoch": epoch,
           # "Accuracy": f"{correct/total_length}: {correct}/{total_length}",
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
    train_dataset = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("datasets", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)

    for epoch in range(1, args.epochs + 1):
        print({"LearningRateLog10": np.log10(optimizer.param_groups[0]["lr"])})
        loss = train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader, epoch)
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