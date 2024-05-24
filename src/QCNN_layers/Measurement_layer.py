import torch


def map_HW_to_measure(batch_x, device):
    return torch.stack([torch.diag(x) for x in batch_x]).to(device)