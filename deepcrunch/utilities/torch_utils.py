import torch
import numpy as np
import random

def set_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)

def run_on_rank_zero(func):
    def wrapper(self, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            return func(self, *args, **kwargs)

    return wrapper