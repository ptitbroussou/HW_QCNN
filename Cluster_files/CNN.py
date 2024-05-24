import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import torch.nn as nn  # the neural network library of pytorch
import load_dataset as load  # module with function to load MNIST


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # first convolutionnal layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=2,
            padding=2
        )

        # first pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second convolutionnal layer
        self.conv2 = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            padding=2
        )

        # second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten
        self.flat = nn.Flatten()

        # fully connected layer, output 10 classes
        self.fc = nn.Linear(162, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        # x = F.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # x = F.relu(x)
        x = self.flat(x)
        output = self.fc(x)
        return output  # return x for visualization


batch_size = 10  # the number of examples per batch
train_loader, test_loader = load.load_MNIST(batch_size=batch_size)
scala = 1000
reduced_loader = load.reduce_MNIST_dataset(train_loader, scala)

conv_network = CNN()
learning_rate = 1e-2  # the scale of the changes applied to the weights
optimizer = torch.optim.Adam(conv_network.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
# test_loss, test_accuracy = eval_net(conv_network, test_loader, criterion)
# print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy*100:.4f} %')

loss_list = []
accuracy_list = []


def train_net(network, train_loader, criterion, optimizer):
    network.train()  # put in train mode: we will modify the weights of the network
    train_loss = 0  # initialize the loss
    train_accuracy = 0  # initialize the accuracy

    # loop on the batches in the train dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # important step to reset gradients to zero

        # Run the network and compute the loss
        output = network(data)  # we run the network on the data
        loss = criterion(output,
                         target)  # we compare output to the target and compute the loss, using the chosen loss function
        train_loss += loss.item()  # we increment the total train loss

        # !!!Here we do the learning!!!
        loss.backward()  # backpropagation: the gradients are automatically computed by the autograd
        optimizer.step()  # specific optimization rule for changing the weights (stochastic gradient descent, Adam etc)
        # and change weighs

        # Getting the prediction of the network and computing the accuracy
        pred = output.argmax(dim=1, keepdim=True)  # the class chosen by the network is the highest output
        acc = pred.eq(target.view_as(pred)).sum().item()  # the accuracy is the proportion of correct classes
        train_accuracy += acc  # increment accuracy of whole test set

    train_accuracy /= len(train_loader.dataset)  # compute mean accuracy
    train_loss /= (batch_idx + 1)  # mean loss
    return train_loss, train_accuracy


for epoch in range(10):
    train_loss, train_accuracy = train_net(conv_network, train_loader, criterion, optimizer)
    loss_list.append(train_loss)
    accuracy_list.append(train_accuracy * 100)

    print(f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %')


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


test_loss, test_accuracy = eval_net(conv_network, test_loader, criterion)
print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
