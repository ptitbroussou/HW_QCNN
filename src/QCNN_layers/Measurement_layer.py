import torch

"""
The measurement operation.
"""


def measurement(batch_density_matrix, device):
    """
    Args:
        - batch_density_matrix: the final density matrices with batch
        - device: torch device (cpu, cuda, etc...)
    Output:
        - The diagonal vectors of input matrices, corresponding to the sampling probability distribution
    """
    return torch.stack([torch.diag(density_matrix) for density_matrix in batch_density_matrix]).to(device)
