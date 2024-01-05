import torch
from torch import nn
from scipy.special import binom
from tqdm import tqdm


def training(model, nbr_epochs, nbr_class, TrainLoader, TestLoader, device):
    """ This function train the model on the data given, according
    to the parameters defined previously. """
    
    lr, betas, eps, weight_decay = 0.001, (0.9, 0.999), 1e-08, 0
    #optimizer = torch.optim.Adam(model.parameters(), lr, betas, eps, weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    
    # Store the loss function evolution through the training:
    list_train_loss = []

    for epoch in range(nbr_epochs):
        running_lass, last_loss = 0, 0
        print("Epoch:{}".format(epoch+1))
        for data in tqdm(TrainLoader, total=len(TrainLoader)):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Define the wanted outputs
            targets = torch.zeros((len(labels), outputs.size()[-2],outputs.size()[-1]))
            for i in range(len(labels)):
                targets[i, labels[i], labels[i]] = 1
            targets = targets.to(device)

            # Compute the loss and its gradients
            print(outputs.size(), targets.size())
            loss = loss_fn(outputs, targets)
            print(loss.item())
            loss.backward()

            # Performing gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            list_train_loss.append(loss.item())
