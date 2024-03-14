import torch


def eval_net(network, test_loader, criterion):
    network.eval()  # put in eval mode: we will just run, not modify the network
    test_loss = 0  # initialize the loss
    test_accuracy = 0  # initialize the accuracy

    with torch.no_grad():  # careful, we do not care about gradients here
        # loop on the batches in the test dataset
        for batch_idx, (data, target) in enumerate(test_loader):
            # Run the network and compute the loss
            output = network(data)  # run the network on the test data
            loss = criterion(output,
                             target)  # compare the output to the target and compute the loss, using the chosen loss function
            test_loss += loss.item()  # increment the total test loss

            # Getting the prediction of the network and computing the accuracy
            pred = output.argmax(dim=1, keepdim=True)  # the class chosen by the network is the highest output
            acc = pred.eq(target.view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
            test_accuracy += acc  # increment accuracy of whole test set

    test_accuracy /= len(test_loader.dataset)  # compute mean accuracy
    test_loss /= (batch_idx + 1)  # mean loss
    return test_loss, test_accuracy



