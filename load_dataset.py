import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def load_FashionMNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def load_MNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def filter_dataloader(dataloader, classes=[0, 1]):
    filtered_data = []
    filtered_targets = []

    for data, target in dataloader:
        mask = target == classes[0]
        for c in classes[1:]:
            mask = mask | (target == c)

        filtered_data.append(data[mask])
        filtered_targets.append(target[mask])

    # Concatenate all collected data and targets
    filtered_data = torch.cat(filtered_data, dim=0)
    filtered_targets = torch.cat(filtered_targets, dim=0)

    # Create a new TensorDataset and DataLoader from the filtered data and targets
    filtered_dataset = TensorDataset(filtered_data, filtered_targets)
    filtered_dataloader = DataLoader(filtered_dataset, batch_size=dataloader.batch_size, shuffle=True)

    return filtered_dataloader


def reduce_MNIST_dataset(data_loader, scala):
    # original data: torch.Size([60000, 28, 28])
    old_data = data_loader.dataset.data
    data_loader.dataset.data = old_data.resize_(int(data_loader.dataset.data.size(0) / scala), 28, 28)
    return data_loader


def to_density_matrix(batch_vectors, device):
    out = torch.zeros([batch_vectors.size(0), batch_vectors.size(1), batch_vectors.size(1)]).to(device)
    index = 0
    for vector in batch_vectors:
        out[index] += torch.einsum('i,j->ij', vector, vector)
        index += 1
    return out


def copy_images_bottom_channel(images, J):
    images = images.unsqueeze(1)
    upscaled_x = F.interpolate(images, size=(images.size()[-1] * J, images.size()[-1] * J), mode='nearest')
    upscaled_x = upscaled_x.squeeze(1)
    return upscaled_x


def copy_images_bottom_channel_stride(images, scale_factor):
    # Assume 'images' is a 3D torch tensor representing a batch of grayscale images
    # Dimension 0 is the batch size, 1 and 2 are both N (rows and columns respectively)
    # 'scale_factor' is the scaling factor

    # Batch and original dimensions
    batch_size, N, _ = images.shape  # Assuming square images N x N

    # New dimensions
    new_N = N * scale_factor

    # Precompute the original indices to be accessed for all images
    orig_i = torch.arange(new_N).floor_divide(scale_factor) % N
    orig_j = torch.arange(new_N).floor_divide(scale_factor) % N

    # Adjust indices to simulate the stride effect by adding a varying offset
    offset = (torch.arange(new_N * new_N).view(new_N, new_N) % scale_factor) * 1
    orig_i = (orig_i.view(-1, 1) + offset) % N
    orig_j = (orig_j.view(1, -1) + offset) % N

    # Use advanced indexing to create the scaled images
    # Apply indexing directly, correctly handling the batch dimension
    scaled_images = images[:, orig_i, orig_j]

    return scaled_images
