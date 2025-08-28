
import torch

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from the_greatest_logging_ever import print, print_lines


name = "Comparative"
description = """
Ask network to determine if two input images are the same. Similar to k-nearest neighbors.
"""

defaults = {
    "batch_size": 32,
    "epochs": 24,
    "learning_rate": 1e-3,
    "log_interval": 100
}

class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, *args, transform=None, **kwargs):
        self.transform = transform
        self.data, self.labels = self.create_data(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index]), self.labels[index]
        else:
            return self.data[index], self.labels[index]

    def create_data(self, dataset):
        """
        Create pairs and is_same=True/False labels.
        Data is output as (batch, pair, height, width), so each of the two images
        in a comparison pair takes up a channel.
        """

        random_permute = lambda x: x[torch.randperm(len(x))]

        # minimum number of samples for digits
        assert len(dataset.classes) == 10
        all_sizes = [len(dataset.data[dataset.targets == label])
                     for label in range(10)]
        size = min(all_sizes)

        # set classes to same size
        all_sets = [{"Same": random_permute(dataset.data[dataset.targets == label])[:size],
                     "Same2": random_permute(dataset.data[dataset.targets == label])[:size],
                     "Different": random_permute(dataset.data[dataset.targets != label])[:size]}
                    for label in range(10)]

        # equal size of same pairs and different pairs
        # half_size = size//2
        # all_sets = [{"SamePairs": torch.stack([set_["Same"][:half_size], set_["Same"][half_size:2*half_size]]).transpose(0, 1),
        #              "DifferentPairs": torch.stack([set_["Same"][:half_size],
        #                                             set_["Different"][half_size:2*half_size]]).transpose(0, 1)}
        #             for set_ in all_sets]
        all_sets = [{"SamePairs": torch.stack([set_["Same"], set_["Same2"]]).transpose(0, 1),
                     "DifferentPairs": torch.stack([set_["Same"],
                                                    set_["Different"]]).transpose(0, 1)}
                    for set_ in all_sets]


        # ensure different pairs are balanced
        def random_flip_pairs(pairs):
            flip_ = torch.randint(2, (len(pairs),), dtype=bool)
            pairs[flip_] = pairs[flip_][:,[1,0]]

            return pairs

        all_sets = [{key: random_flip_pairs(value)
                    for key, value in dict_.items()}
                    for dict_ in all_sets]

        # random mix of same pairs and different pairs
        def random_combine(array1, array2):
            assert len(array1) == len(array2)
            combined = torch.concat([array1, array2])
            permutation = torch.randperm(len(combined))
            combined = combined[permutation]

            labels = permutation < len(array1)

            return combined, labels

        all_sets = [random_combine(dict_["SamePairs"], dict_["DifferentPairs"])
                    for dict_ in all_sets]

        # shuffle all data
        all_data = torch.concat([data for data, labels in all_sets])
        all_labels = torch.concat([labels for data, labels in all_sets])

        permutation = torch.randperm(len(all_data))
        all_data = all_data[permutation]
        all_labels = all_labels[permutation].long()

        return all_data, all_labels

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=2,
                      out_channels=32,
                      kernel_size=(3, 3),
                      groups=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3, 3),
                      groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),
            nn.Dropout2d(0.15),

            nn.Flatten(),

            nn.Linear(in_features=9216, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=2),
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
    
    print({"LossesAverage": sum(losses) / len(losses)})
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
           "TestSetAverageLoss": test_loss / len(test_loader.dataset)})


def run(**kwargs):
    args = argparse.Namespace(**(defaults | kwargs))
    print({"Arguments": args.__dict__})

    device = torch.accelerator.current_accelerator()

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.batch_size}

    accel_kwargs = {"shuffle": True}

    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        lambda x: x.float() / 255,
        ])
    train_dataset = PairsDataset(datasets.MNIST("datasets", train=True, download=True), transform=transform)
    test_dataset = PairsDataset(datasets.MNIST("datasets", train=False), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(1, args.epochs + 1):
        print({"LearningRateLog10": np.log10(optimizer.param_groups[0]["lr"])})
        loss = train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step(metrics=loss)

    return model

if __name__ == "__main__":
    run(**defaults)