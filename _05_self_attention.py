
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


name = "Self-attention"
description = """
Self-attention at the pixel level.
"""

defaults = {
    "batch_size": 64,
    "epochs": 14,
    "learning_rate": 1e-4,
    "log_interval": 200
}

class SelfAttention(nn.Module):
    def __init__(self,
                 embedding_dimension,
                 key_dimension,
                 value_dimension):
        super().__init__()

        self.key_dimension = key_dimension

        self.keys = nn.Linear(embedding_dimension, key_dimension, bias=False)
        self.queries = nn.Linear(embedding_dimension, key_dimension, bias=False)
        self.values = nn.Linear(embedding_dimension, value_dimension, bias=False)
        
    def forward(self, x):
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        # x = queries @ keys.transpose(-2, -1) / self.key_dimension**0.5
        # x = torch.softmax(x, dim=-1) @ values
        x = nn.functional.scaled_dot_product_attention(queries, keys, values, dropout_p=-1)

        return x

class Residual(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return x + self.network(x)

class Identity(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x)

class LambdaModule(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

class Model(nn.Module):
    def __init__(self,
                 num_discrete_classes=8,
                 embedding_dimension=8,
                 key_dimension=4,
                 value_dimension=16,
                 num_blocks=2):
        super(Model, self).__init__()

        self.key_dimension = key_dimension
        image_size = 28*28
        self.embedding = nn.Embedding(num_discrete_classes, embedding_dimension)
        self.positional_embedding = nn.Embedding(image_size, embedding_dimension)

        # block input and output shape is: batch, image_size, embedding_dimension
        block_make = lambda block_index: nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.LayerNorm(normalized_shape=(image_size, embedding_dimension)),
                    *[LambdaModule(lambda x: x + self.positional_embedding.weight)
                      for _ in range(block_index == 0)],
                    SelfAttention(embedding_dimension=embedding_dimension,
                                  key_dimension=key_dimension,
                                  value_dimension=value_dimension),
                    nn.Flatten(),
                    nn.Linear(image_size*value_dimension, image_size*embedding_dimension),
                    LambdaModule(lambda x: x.reshape(-1, image_size, embedding_dimension)))),
            Residual(
                nn.Sequential(
                    nn.Flatten(),
                    nn.LayerNorm(normalized_shape=image_size*embedding_dimension),
                    nn.Linear(image_size*embedding_dimension, image_size*embedding_dimension),
                    nn.ReLU(),
                    nn.Linear(image_size*embedding_dimension, image_size*embedding_dimension),
                    LambdaModule(lambda x: x.reshape(-1, image_size, embedding_dimension)))))

        self.sequence = nn.Sequential(
            nn.Flatten(),
            LambdaModule(lambda x: x // (256 // num_discrete_classes)),
            self.embedding,
            *(block_make(index)
              for index in range(num_blocks)),
            nn.Flatten(),
            nn.Linear(image_size*embedding_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10)
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

    transform = lambda x: torch.tensor(np.array(x, dtype=np.long))
    train_dataset = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("datasets", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

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