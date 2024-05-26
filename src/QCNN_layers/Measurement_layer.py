import torch


def measurement(batch_x, device):
    return torch.stack([torch.diag(x) for x in batch_x]).to(device)