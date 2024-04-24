import torch
from torch import nn
from scipy.special import binom
from torch.nn import AdaptiveAvgPool2d
import torch.nn.functional as F
from toolbox import to_density_matrix, copy_images_bottom_channel, map_HW_to_measure


# from tqdm import tqdm
#
#
# def training(model, nbr_epochs, nbr_class, TrainLoader, TestLoader, device):
#     """ This function train the model on the data given, according
#     to the parameters defined previously. """
#
#     lr, betas, eps, weight_decay = 0.001, (0.9, 0.999), 1e-08, 0
#     #optimizer = torch.optim.Adam(model.parameters(), lr, betas, eps, weight_decay)
#     optimizer = torch.optim.SGD(model.parameters(), 0.01)
#     #loss_fn = nn.CrossEntropyLoss()
#     loss_fn = nn.MSELoss()
#
#     # Store the loss function evolution through the training:
#     list_train_loss = []
#
#     for epoch in range(nbr_epochs):
#         running_lass, last_loss = 0, 0
#         print("Epoch:{}".format(epoch+1))
#         for data in tqdm(TrainLoader, total=len(TrainLoader)):
#             # Every data instance is an input + label pair
#             inputs, labels = data
#             inputs = inputs.to(device)
#
#             # Zero your gradients for every batch!
#             optimizer.zero_grad()
#
#             # Make predictions for this batch
#             outputs = model(inputs)
#
#             # Define the wanted outputs
#             targets = torch.zeros((len(labels), outputs.size()[-2],outputs.size()[-1]))
#             for i in range(len(labels)):
#                 targets[i, labels[i], labels[i]] = 1
#             targets = targets.to(device)
#
#             # Compute the loss and its gradients
#             print(outputs.size(), targets.size())
#             loss = loss_fn(outputs, targets)
#             print(loss.item())
#             loss.backward()
#
#             # Performing gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#
#             # Adjust learning weights
#             optimizer.step()
#
#             # Gather data and report
#             list_train_loss.append(loss.item())

def train_net(batch_size, I, J, k, network, train_loader, criterion, optimizer, device):
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # important step to reset gradients to zero

        new_size = I
        adaptive_avg_pool = AdaptiveAvgPool2d((new_size, new_size))
        data = adaptive_avg_pool(data).to(device)
        init_density_matrix = to_density_matrix(F.normalize(data.squeeze().resize(batch_size,I**2), p=2, dim=1).to(device), device)
        channel_data = copy_images_bottom_channel(init_density_matrix, J).to(device)
        output = network(channel_data)  # we run the network on the data

        # training
        loss = criterion(output,target.to(device))  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        loss.backward()
        optimizer.step()

        # predict
        pred = output.argmax(dim=1, keepdim=True).to(device)  # the class chosen by the network is the highest output
        acc = pred.eq(target.to(device).view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy


def test_net(batch_size, I, J, k, network, train_loader, criterion, device):
    # network.eval()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for batch_idx, (data, target) in enumerate(train_loader):
        # Run the network and compute the loss
        new_size = I
        adaptive_avg_pool = AdaptiveAvgPool2d((new_size, new_size))
        data = adaptive_avg_pool(data).to(device)
        init_density_matrix = to_density_matrix(F.normalize(data.squeeze().resize(batch_size,I**2), p=2, dim=1).to(device), device)
        channel_data = copy_images_bottom_channel(init_density_matrix, J).to(device)
        output = network(channel_data)  # we run the network on the data

        loss = criterion(output,target.to(device))  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(dim=1, keepdim=True)  # the class chosen by the network is the highest output
        acc = pred.eq(target.to(device).view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy