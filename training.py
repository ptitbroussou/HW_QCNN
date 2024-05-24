from torch.nn import AdaptiveAvgPool2d
import torch.nn.functional as F

from load_dataset import to_density_matrix, copy_images_bottom_channel, copy_images_bottom_channel_stride


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


def train_net_stride(batch_size, I, J, k, network, train_loader, criterion, optimizer, device):
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
        channel_data = copy_images_bottom_channel_stride(init_density_matrix, J).to(device)
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


def test_net_stride(batch_size, I, J, k, network, train_loader, criterion, device):
    # network.eval()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for batch_idx, (data, target) in enumerate(train_loader):
        # Run the network and compute the loss
        new_size = I
        adaptive_avg_pool = AdaptiveAvgPool2d((new_size, new_size))
        data = adaptive_avg_pool(data).to(device)
        init_density_matrix = to_density_matrix(F.normalize(data.squeeze().resize(batch_size,I**2), p=2, dim=1).to(device), device)
        channel_data = copy_images_bottom_channel_stride(init_density_matrix, J).to(device)
        output = network(channel_data)  # we run the network on the data

        loss = criterion(output,target.to(device))  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(dim=1, keepdim=True)  # the class chosen by the network is the highest output
        acc = pred.eq(target.to(device).view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy