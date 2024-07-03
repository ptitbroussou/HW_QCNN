import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import torch
from torch.nn import AdaptiveAvgPool2d
from src.load_dataset import load_fashion_mnist
import torch.nn.functional as F


def load_image_states_fashion_mnist(I, class_set, train_dataset_number, test_dataset_number, device):
    """ This function loads the image from the Fashion MNIST dataset and preprocess
    them so it can be seen as a normalized tensor encoded state of dimension I**2.
    Args:
        - I: dimension of each image
        - class_set: list of classes to consider (integers between 0 and 9)
        - train_dataset_number: number of training images to consider
        - test_dataset_number: number of testing images to consider
    Output:
        - train_init_states: tensor of shape (train_dataset_number, I**2) containing the
        training states
        - test_init_states: tensor of shape (test_dataset_number, I**2) containing the
        testing states
    """
    train_init_states, test_init_states = torch.zeros(train_dataset_number, I ** 2), torch.zeros(test_dataset_number, I ** 2)
    train_dataloader, test_dataloader = load_fashion_mnist(class_set, train_dataset_number, test_dataset_number, batch_size=1)
    train_labels, test_labels = torch.zeros(train_dataset_number), torch.zeros(test_dataset_number)
    adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
    print("Loading training images and labels:")
    pbar_1 = tqdm(total = train_dataset_number)
    for i, (data, label) in enumerate(train_dataloader):
        data = adaptive_avg_pool(data).to(device)
        train_init_states[i] = F.normalize(data.squeeze().resize(data.shape[0], I ** 2), p=2, dim=1)
        train_labels[i] = label
        pbar_1.update(1)
    pbar_1.close()
    print("Loading testing images and labels:")
    pbar_2 = tqdm(total = test_dataset_number)
    for i, (data, label) in enumerate(test_dataloader):
        data = adaptive_avg_pool(data).to(device)
        test_init_states[i] = F.normalize(data.squeeze().resize(data.shape[0], I ** 2), p=2, dim=1)
        test_labels[i] = label
        pbar_2.update(1)
    pbar_2.close()
    return train_init_states, test_init_states, train_labels, test_labels