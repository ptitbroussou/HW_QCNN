import torch
from torchvision import datasets, transforms
from sklearn import datasets as skdatasets

transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def load_FashionMNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=True, download=True, transform=transform_mnist),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, download=True, transform=transform_mnist),
            batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, 784, 10


def load_MNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform_mnist),
            batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=transform_mnist),
            batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 784, 10


class DigitsDataset(torch.utils.data.Dataset):
    """
    create a dataset of datas_number point with 1/4 for test_dataset
    """
    def __init__(self, train=True,  n_class=10):
        X, y = skdatasets.load_digits(n_class=n_class, return_X_y=True)        
        """
        torch.randperm is meant to shuffle the dataset to avoid having always the same combination train/test
        """
        idx = torch.randperm(len(y))
        X, y = X[idx], y[idx]
        if train:
            self.data, self.targets = X[:int(len(X) / 4 * 3)] / X.max(), y[:int(len(X) / 4 * 3)]
        else:
            self.data, self.targets = X[int(len(X) / 4 * 3):] / X.max(), y[int(len(X) / 4 * 3):]         
        self.data = torch.FloatTensor(self.data)
        self.targets = torch.LongTensor(self.targets)
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.data.size(0)


def load_digits(batch_size):
    """
    create loaders of digits dataset

    Attributs:
    ==========
        datas_number : int, number of data
        batch_size : int, the size of batch

    Returns:
    ========
        train_loader, test_loader : the data loader
    """
    train_set = DigitsDataset(train=True, n_class=10)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = DigitsDataset(train=False, n_class=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        
    return train_loader, test_loader, 64, 10


class CirclesDataset(torch.utils.data.Dataset):
    """
    create a dataset of datas_number point with 1/4 for test_dataset
    """
    def __init__(self, train=True, n_samples=1000, shuffle=True, noise=0.1, random_state=None, factor=0.5):
        X, y = skdatasets.make_circles(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state, factor=factor) 

        if train:
            self.data, self.targets = X[:int(len(X) / 4 * 3)] / X.max(), y[:int(len(X) / 4 * 3)]
        else:
            self.data, self.targets = X[int(len(X) / 4 * 3):] / X.max(), y[int(len(X) / 4 * 3):]  
        self.data = torch.FloatTensor(self.data)
        self.targets = torch.LongTensor(torch.from_numpy(self.targets))
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.data.size(0)


def load_circles(batch_size, train=True, n_samples=1000, shuffle=True, noise=0.1, random_state=None, factor=0.5):

    train_set = CirclesDataset(train=True, n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state, factor=factor)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = CirclesDataset(train=False, n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state, factor=factor)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        
    return train_loader, test_loader, 2, 1

class MoonsDataset(torch.utils.data.Dataset):
    """
    create a dataset of datas_number point with 1/4 for test_dataset
    """
    def __init__(self, train=True, n_samples=1000, shuffle=True, noise=0.1, random_state=None, factor=0.5):
        X, y = skdatasets.make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state) 

        if train:
            self.data, self.targets = X[:int(len(X) / 4 * 3)] / X.max(), y[:int(len(X) / 4 * 3)]
        else:
            self.data, self.targets = X[int(len(X) / 4 * 3):] / X.max(), y[int(len(X) / 4 * 3):]         
        self.data = torch.FloatTensor(self.data)
        self.targets = torch.LongTensor(torch.from_numpy(self.targets))
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.data.size(0)


def load_moons(batch_size, train=True, n_samples=1000, shuffle=True, noise=0.1, random_state=None):

    train_set = MoonsDataset(train=True, n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = MoonsDataset(train=False, n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        
    return train_loader, test_loader, 2, 1