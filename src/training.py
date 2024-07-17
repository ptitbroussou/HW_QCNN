import time

import torch
from torch.nn import AdaptiveAvgPool2d
import torch.nn.functional as F

from src.load_dataset import to_density_matrix, copy_images_bottom_channel_stride, RGB_images_to_density_matrix
from src.toolbox import normalize_DM


def train_network(batch_size, I, J, network, train_loader, criterion, output_scale, optimizer, stride, device):
    """
    Train the single-channel image dataset once
    """
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # important step to reset gradients to zero
        adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
        # preprocess, pooling operation classically
        data = adaptive_avg_pool(data).to(device)
        data = data.sum(dim=1, keepdim=True)
        # vectorize the image matrix, and then inner product itself to obtain the density matrix
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], I ** 2), p=2, dim=1).to(device), device)
        # Add J channels to the original image, the size of the density matrix becomes J times the original size, then normalize it.
        channel_data = normalize_DM(copy_images_bottom_channel_stride(init_density_matrix, J, stride)).to(device)
        output = network(channel_data)  # we run the network on the data
        # training
        loss = criterion(output*output_scale, target.to(
            device))  # we compare output to the target and compute the loss, using the chosen loss function with output_scale
        train_loss += loss.item()  # we increment the total train loss
        loss.backward(retain_graph=True)
        optimizer.step()

        # predict
        pred = output.argmax(dim=1, keepdim=True).to(device)  # the class chosen by the network is the highest output
        acc = pred.eq(target.to(device).view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy


def test_network(batch_size, I, J, network, test_loader, criterion, output_scale, stride, device):
    """
    Test the single-channel image dataset once
    """
    network.eval()  # put in eval mode: we will not modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for batch_idx, (data, target) in enumerate(test_loader):
        # Run the network and compute the loss
        adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
        # preprocess, pooling operation classically
        data = adaptive_avg_pool(data).to(device)
        data = data.sum(dim=1, keepdim=True)
        # vectorize the image matrix, and then inner product itself to obtain the density matrix
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], I ** 2), p=2, dim=1).to(device), device)
        # Add J channels to the original image, the size of the density matrix becomes J times the original size, then normalize it.
        channel_data = normalize_DM(copy_images_bottom_channel_stride(init_density_matrix, J, stride)).to(device)
        output = network(channel_data)  # we run the network on the data

        loss = criterion(output*output_scale, target.to(
            device))  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(dim=1, keepdim=True)  # the class chosen by the network is the highest output
        acc = pred.eq(target.to(device).view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(test_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy


def train_RGB_network(batch_size, I, J, network, train_loader, criterion, output_scale, optimizer, stride, device):
    """
    Train the multi-channel image dataset once
    """
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        if target.dim() == 2:
            target = target.squeeze(1)
        optimizer.zero_grad()  # important step to reset gradients to zero
        adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
        # preprocess, pooling operation classically
        data = adaptive_avg_pool(data).to(device)
        # vectorize the image matrix, and then inner product itself to obtain the density matrix, convert 3 channels to 1 channel.
        data = RGB_images_to_density_matrix(I, data, device)
        # Add J channels to the original image, the size of the density matrix becomes J times the original size, then normalize it.
        data = normalize_DM(copy_images_bottom_channel_stride(data, int(J / 3), stride))
        output = network(data)  # we run the network on the data

        # training
        loss = criterion(output_scale*output, target.to(
            device))  # we compare output to the target and compute the loss, using the chosen loss function
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


def test_RGB_network(batch_size, I, J, network, test_loader, criterion, output_scale, stride, device):
    """
    Test the multi-channel image dataset once
    """
    network.eval()  # put in eval mode: we will not modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for batch_idx, (data, target) in enumerate(test_loader):
        if target.dim() == 2:
            target = target.squeeze(1)
        # Run the network and compute the loss
        adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
        # preprocess, pooling operation classically
        data = adaptive_avg_pool(data).to(device)
        # vectorize the image matrix, and then inner product itself to obtain the density matrix, convert 3 channels to 1 channel.
        data = RGB_images_to_density_matrix(I, data, device)
        # Add J channels to the original image, the size of the density matrix becomes J times the original size, then normalize it.
        data = copy_images_bottom_channel_stride(data, int(J / 3), stride)
        data = normalize_DM(data)
        output = network(data)  # we run the network on the data

        loss = criterion(output_scale*output, target.to(device))  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(dim=1, keepdim=True)  # the class chosen by the network is the highest output
        acc = pred.eq(target.to(device).view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(test_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy


def train_RGB_globally(batch_size, I, J, network, reduced_train_loader, reduced_test_loader, optimizer, scheduler,
                        criterion, output_scale, train_epochs, test_interval, stride, device):
    """
    Perform general training on the multichannel image network, including training, testing, and saving data.
    """
    # print number of parameters of network
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Start training! Number of network total parameters: {total_params}")

    # test before training
    test_loss, test_accuracy = test_RGB_network(batch_size, I, J, network, reduced_test_loader, criterion, output_scale, stride, device)
    print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')

    # training step
    loss_list = []
    accuracy_list = []
    for epoch in range(train_epochs):
        start = time.time()
        train_loss, train_accuracy = train_RGB_network(batch_size, I, J, network, reduced_train_loader, criterion,
                                                       output_scale, optimizer, stride, device)
        loss_list.append(train_loss)
        accuracy_list.append(train_accuracy * 100)
        end = time.time()
        print(
            f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %, time={(end - start):.4f}s')
        if epoch % test_interval == 0 and epoch != 0:
            test_loss, test_accuracy = test_RGB_network(batch_size, I, J, network, reduced_test_loader, criterion, output_scale, stride,
                                                        device)
            print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
        scheduler.step()

    # test after training
    test_loss, test_accuracy = test_RGB_network(batch_size, I, J, network, reduced_test_loader, criterion, output_scale, stride, device)
    print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')

    # saving data
    return network.state_dict()


def train_globally(batch_size, I, J, network, reduced_train_loader, reduced_test_loader, optimizer, scheduler,
                   criterion, output_scale, train_epochs, test_interval, stride, device):
    """
   Perform general training on the single channel image network, including training, testing, and saving data.
   """
    # print number of parameters of network
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Start training! Number of network total parameters: {total_params}")

    # test before training
    test_loss, test_accuracy = test_network(batch_size, I, J, network, reduced_test_loader, criterion, output_scale, stride, device)
    print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')

    # training step
    training_loss_list = []
    testing_loss_list = []
    training_accuracy_list = []
    testing_accuracy_list = []
    for epoch in range(train_epochs):
        start = time.time()
        train_loss, train_accuracy = train_network(batch_size, I, J, network, reduced_train_loader, criterion, output_scale,
                                                   optimizer, stride, device)
        training_loss_list.append(train_loss)
        training_accuracy_list.append(train_accuracy * 100)
        end = time.time()
        print(
            f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %, time={(end - start):.4f}s')
        if (epoch+1) % test_interval == 0:
            test_loss, test_accuracy = test_network(batch_size, I, J, network, reduced_test_loader, criterion, output_scale, stride,
                                                    device)
            testing_loss_list.append(test_loss)
            testing_accuracy_list.append(test_accuracy)
            print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
            # print(os.getcwd())
            # torch.save(network.state_dict(), "Model_states/model" + str(epoch))  # save network parameters
        scheduler.step()

    # saving data
    return network.state_dict(), training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list


def train_network_2D(batch_size, I, network, train_loader, criterion, output_scale, optimizer, device):
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # important step to reset gradients to zero
        adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
        data = adaptive_avg_pool(data).to(device)
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], I ** 2), p=2, dim=1).to(device), device)
        output = network(normalize_DM(init_density_matrix))  # we run the network on the data

        # training
        # print(output)
        # print(target)
        loss = criterion(output*output_scale, target.to(device))  # we compare output to the target and compute the loss, using the chosen loss function
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


def test_network_2D(batch_size, I, network, test_loader, criterion, output_scale, device):
    network.eval()  # put in eval mode: we will not modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy
    for batch_idx, (data, target) in enumerate(test_loader):
        # Run the network and compute the loss
        adaptive_avg_pool = AdaptiveAvgPool2d((I, I))
        data = adaptive_avg_pool(data).to(device)
        init_density_matrix = to_density_matrix(
            F.normalize(data.squeeze().resize(data.shape[0], I ** 2), p=2, dim=1).to(device), device)
        output = network(normalize_DM(init_density_matrix))  # we run the network on the data
        loss = criterion(output*output_scale, target.to(device))  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss
        pred = output.argmax(dim=1, keepdim=True)  # the class chosen by the network is the highest output
        acc = pred.eq(target.to(device).view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(test_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy


def train_globally_2D(batch_size, I, network, reduced_train_loader, reduced_test_loader, optimizer, scheduler,
                      criterion, output_scale, train_epochs, test_interval, device):
    """
    Perform general training on the single channel image and single channel network, including training, testing, and saving data.
    """
    # print number of parameters of network
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Start training! Number of network total parameters: {total_params}")

    test_loss, test_accuracy = test_network_2D(batch_size, I, network, reduced_test_loader, criterion, output_scale, device)
    print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')

    # training step
    training_loss_list = []
    testing_loss_list = []
    training_accuracy_list = []
    testing_accuracy_list = []
    for epoch in range(train_epochs):
        start = time.time()
        train_loss, train_accuracy = train_network_2D(batch_size, I, network, reduced_train_loader, criterion, output_scale,
                                                      optimizer, device)
        training_loss_list.append(train_loss)
        training_accuracy_list.append(train_accuracy * 100)
        end = time.time()
        print(
            f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %, time={(end - start):.4f}s')
        if epoch % test_interval == 0 and epoch != 0:
            test_loss, test_accuracy = test_network_2D(batch_size, I, network, reduced_test_loader, criterion, output_scale, device)
            testing_loss_list.append(test_loss)
            testing_accuracy_list.append(test_accuracy)
            print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
        scheduler.step()
    # final testing part
    test_loss, test_accuracy = test_network_2D(batch_size, I, network, reduced_test_loader, criterion, output_scale, device)
    print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
    return network.state_dict(), training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list
