import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO
import torch.nn.functional as F
from sklearn.decomposition import PCA

transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def load_mnist(class_set, train_dataset_number, test_dataset_number, batch_size):
    filtered_dataset = load_filtered_mnist(class_set, train_dataset_number, train=True)
    train_dataloader = create_dataloader(filtered_dataset, batch_size)
    filtered_test_dataset = load_filtered_mnist(class_set, test_dataset_number, train=False)
    test_dataloader = create_dataloader(filtered_test_dataset, batch_size)
    return train_dataloader, test_dataloader


def load_fashion_mnist(class_set, train_dataset_number, test_dataset_number, batch_size):
    filtered_dataset = load_filtered_fashion_mnist(class_set, train_dataset_number, train=True)
    train_dataloader = create_dataloader(filtered_dataset, batch_size)
    filtered_test_dataset = load_filtered_fashion_mnist(class_set, test_dataset_number, train=False)
    test_dataloader = create_dataloader(filtered_test_dataset, batch_size)
    return train_dataloader, test_dataloader


def load_medmnist(dataset_name, class_set, train_dataset_number, test_dataset_number, batch_size):
    filtered_train_dataset = load_filtered_medmnist(dataset_name, class_set, train_dataset_number, split='train')
    train_dataloader = create_dataloader(filtered_train_dataset, batch_size)
    filtered_test_dataset = load_filtered_medmnist(dataset_name, class_set, test_dataset_number, split='test')
    test_dataloader = create_dataloader(filtered_test_dataset, batch_size)
    return train_dataloader, test_dataloader


def load_cifar10(class_set, train_dataset_number, test_dataset_number, batch_size):
    filtered_train_dataset = load_filtered_cifar10(class_set, train_dataset_number, train=True)
    train_dataloader = create_dataloader(filtered_train_dataset, batch_size)
    filtered_test_dataset = load_filtered_cifar10(class_set, test_dataset_number, train=False)
    test_dataloader = create_dataloader(filtered_test_dataset, batch_size)
    return train_dataloader, test_dataloader


def load_filtered_medmnist(dataset_name, labels_to_include, num_datapoints, split='train'):
    # Get dataset information
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load the MedMNIST dataset
    medmnist_dataset = DataClass(split=split, transform=transform, download=True)

    # Find indices of the datapoints that match the labels_to_include
    indices = [i for i, (img, label) in enumerate(medmnist_dataset) if label in labels_to_include]

    # Shuffle indices to get a random subset
    random_indices = torch.randperm(len(indices)).tolist()

    # Select the specified number of datapoints
    selected_indices = indices[:num_datapoints]

    # Create a subset of the dataset
    filtered_dataset = Subset(medmnist_dataset, selected_indices)

    return filtered_dataset


def load_filtered_fashion_mnist(labels_to_include, num_datapoints, train=True):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the FashionMNIST dataset
    fashion_mnist = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)

    # Find indices of the datapoints that match the labels_to_include
    indices = [i for i, (img, label) in enumerate(fashion_mnist) if label in labels_to_include]

    # Shuffle indices to get a random subset
    random_indices = torch.randperm(len(indices)).tolist()

    # Select the specified number of datapoints
    selected_indices = indices[:num_datapoints]

    # Create a subset of the dataset
    filtered_dataset = Subset(fashion_mnist, selected_indices)

    return filtered_dataset


def load_filtered_mnist(labels_to_include, num_datapoints, train=True):
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    mnist = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    # Find indices of the datapoints that match the labels_to_include
    indices = [i for i, (img, label) in enumerate(mnist) if label in labels_to_include]

    # Shuffle indices to get a random subset
    random_indices = torch.randperm(len(indices)).tolist()

    # Select the specified number of datapoints
    selected_indices = indices[:num_datapoints]

    # Create a subset of the dataset
    filtered_dataset = Subset(mnist, selected_indices)

    return filtered_dataset


def load_filtered_cifar10(labels_to_include, num_datapoints, train=True):
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load the CIFAR-10 dataset
    cifar10 = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

    # Find indices of the datapoints that match the labels_to_include
    indices = [i for i, (img, label) in enumerate(cifar10) if label in labels_to_include]

    # Shuffle indices to get a random subset
    random_indices = torch.randperm(len(indices)).tolist()

    # Select the specified number of datapoints
    selected_indices = indices[:num_datapoints]

    # Create a subset of the dataset
    filtered_dataset = Subset(cifar10, selected_indices)

    return filtered_dataset


# Function to create a DataLoader for the filtered dataset
def create_dataloader(filtered_dataset, batch_size=32):
    dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_FashionMNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def load_MNIST(batch_size, shuffle):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=shuffle)
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


def apply_pca(dataloader, pca=None, new_image_size=10, fit=True):
    """
    Apply PCA to the DataLoader. If fit=True, the PCA model is fitted to the data.
    If fit=False, the data is only transformed based on the provided PCA model.

    Args:
        dataloader (DataLoader): The input DataLoader, with data and labels.
        pca (PCA): Pre-fitted PCA model (used for the test set).
        n_components (int): Number of principal components to keep (for fitting).
        fit (bool): If True, fit the PCA model on the dataset. If False, only transform the data.

    Returns:
        DataLoader: A new DataLoader with PCA applied to the data.
        PCA: The fitted PCA model (if fit=True).
    """
    # Accumulate all data and labels
    data_list = []
    label_list = []

    for data, labels in dataloader:
        # Flatten the data if it's not already flattened
        if len(data.shape) > 2:
            # data = data.sum(dim=1, keepdim=True)
            data = data.view(data.size(0), -1)

        data_list.append(data)
        label_list.append(labels)

    # Stack all data into a single tensor
    all_data = torch.cat(data_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    # Convert to numpy array for PCA
    all_data_np = all_data.numpy()

    # If we are fitting PCA (on the training data)
    if fit:
        pca = PCA(n_components=new_image_size**2)
        all_data_pca = pca.fit_transform(all_data_np)  # Fit PCA on the training data
    else:
        all_data_pca = pca.transform(all_data_np)  # Transform based on pre-fitted PCA (test data)

    # Convert back to torch tensor and reshape to new image size
    all_data_pca_tensor = torch.tensor(all_data_pca, dtype=torch.float32)
    all_data_pca_tensor = all_data_pca_tensor.view(all_data_pca_tensor.shape[0],1,new_image_size, new_image_size)

    # Create a new dataset and DataLoader
    pca_dataset = TensorDataset(all_data_pca_tensor, all_labels)
    pca_dataloader = DataLoader(pca_dataset, batch_size=dataloader.batch_size, shuffle=False)

    return pca_dataloader, pca


def reduce_MNIST_dataset(data_loader, dataset, is_train):
    # original data: torch.Size([60000, 28, 28])
    if is_train:
        scala = 60000//dataset
    else:
        scala = 10000//dataset
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


def copy_images_bottom_channel_stride(images, scale_factor, stride):
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
    offset = (torch.arange(new_N * new_N).view(new_N, new_N) % scale_factor) * stride
    orig_i = (orig_i.view(-1, 1) + offset) % N
    orig_j = (orig_j.view(1, -1) + offset) % N

    # Use advanced indexing to create the scaled images
    # Apply indexing directly, correctly handling the batch dimension
    scaled_images = images[:, orig_i, orig_j]

    return scaled_images


def RGB_images_to_density_matrix(I, data, device):
    """
    vectorize the image matrix, and then inner product itself to obtain the density matrix,
    finally convert the 4-dimensional tensor to the 3-dimensional tesnor, i.e. 3 channels to 1 channel
    Args:
        - I: size of image
        - data: input data from dataset, shape: batch * channels(by default is 3) * I * I
        - device: torch device (cpu, cuda, etc...)
    Output: encoded density matrix, shape: batch * (I^2*channels) * (I^2*channels)
    """
    channels = data.shape[1]
    data = F.normalize(data.resize(data.shape[0],channels,I**2), p=2, dim=2).to(device)
    new_tensor = torch.zeros((data.shape[0], channels*I*I), device=device, dtype=torch.float32)
    for b in range(data.shape[0]):
        for i in range(I**2):
            for c in range(channels):
                new_tensor[b,channels*i+c] = data[b,c,i]
    return to_density_matrix(new_tensor, device)
