
"""
To promote logs that are clearer and more easily processed (no need to use regex)
and resilient to changes, this module encourages the use of JSON for logging.
"""

import json
import random

import numpy as np
import torch

import collections.abc

def awesomize(logging_function):
    def new_logging_function(item, *args, **kwargs):
        assert isinstance(item, dict), f"Expected dictionary type, got {type(item)}."
        json_line = json.dumps(item, default=default_json)
        logging_function(json_line, *args, **kwargs)

    return new_logging_function

def default_json(item):
    match item:
        case np.ndarray(data=data) | torch.Tensor(data=data):
            return data.tolist()
        case collections.abc.Sequence():
            return list(item)
        case other:
            return str(other)


def print_lines(item, key: tuple = (0,), label="x", sample_only=True):
    match item:
        case torch.Tensor(data=t):
            indices = "][".join(map(str, key))
            print({f"{label}.shape": item.shape,
                   f"{label}[{indices}].shape": item[key].shape,
                   "GradientFunction": item.grad_fn and item.grad_fn.name() or None})

            if sample_only and len(item[key]) > 5:
                indices = list(range(2)) + sorted(random.sample(range(2, len(item[key])), 2))
            else:
                indices = range(len(item[key]))

            for index in indices:
                indices = "][".join(map(str, (*key, index)))
                print({f"{label}[{indices}]": item[key][index]})

def summary(item, label=None):
    label = label or "x"
    key = (0,)
    match item:
        case torch.Tensor(data=t):
            indices = "][".join(map(str, key))
            print({f"{label}.shape": item.shape,
                   f"{label}[{indices}].shape": item[key].shape,
                   "GradientFunction": item.grad_fn and item.grad_fn.name() or None})


print = awesomize(print)
