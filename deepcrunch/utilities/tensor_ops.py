import numpy as np


def compute_sparsity(tensor: np.ndarray) -> tuple:
    """Compute the sparsity.

    Args:
        tensor: Pytorch tensor

    Return:
        (the original tensor size, number of zero elements, number of non-zero elements)
    """
    tensor_size = np.prod(tensor.shape)
    dense_size = np.count_nonzero(tensor)
    return tensor_size, tensor_size - dense_size, dense_size
