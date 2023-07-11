import random
import os
import uuid
from typing import Callable

import numpy as np
import torch
from torch import nn

FILE_SIZE_IDENTIFIER = {
    "KB": 1e3,
    "MB": 1e6,
    "GB": 1e9,
    "TB": 1e12,
}

def set_seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)


def run_on_rank_zero(func: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            return func(self, *args, **kwargs)

    return wrapper

def get_uid() -> str:
    return str(uuid.uuid4())

def get_temp_file_name() -> str:
    return f"tmp_{get_uid()}.pt"

def get_number_from_size_identifier(size_identifier: str = "MB") -> int:
    size_identifier = size_identifier.upper()
    assert size_identifier in FILE_SIZE_IDENTIFIER, f"Invalid size identifier {size_identifier}."
    return FILE_SIZE_IDENTIFIER[size_identifier]

def get_model_size(model: nn.Module, size_identifier: str = "MB") -> int:
    size_unit = get_number_from_size_identifier(size_identifier)
    tmp_file_name = get_temp_file_name()
    torch.save(model.state_dict(), tmp_file_name)
    model_size = os.path.getsize(tmp_file_name)//size_unit
    os.remove(tmp_file_name)
    return model_size

# Calculate the approximated model size by it's parameter and quantized bit size
def get_approx_model_size_by_bit(model: nn.Module, bit: int = 8, size_identifier: str = "MB") -> int:
    size_unit = get_number_from_size_identifier(size_identifier)
    model_size = count_parameters(model) / size_unit * bit / 8

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


# Count total model parameters for both trainable and non-trainable
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())