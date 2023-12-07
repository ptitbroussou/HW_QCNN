import torch
from torch import nn
from scipy.special import binom
from tqdm import tqdm


def training(model, nbr_epochs, TrainLoader, Test_Loader, device):
    """ This function train the model on the data given, according
    to the parameters defined previously. """
    for epoch in range(nbr_epochs):
        running_lass, last_loss = 0, 0
        print("Epoch:{}".format(epoch+1))
        for data in tqdm(TrainLoader, total=len(TrainLoader)):
            # Every data instance is an input + label pair
            inputs, labels = data
            targets = torch.zeros((len(labels), nbr_class))
            for i in range(len(labels)):
                targets[i, labels[i]] = 1
            #print(labels)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            print(outputs, targets)
            loss = loss_fn(outputs, targets)
            print(loss.item())
            loss.backward()

            # Performing gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            list_train_loss.append(loss.item())
