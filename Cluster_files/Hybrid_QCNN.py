import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.simplefilter('ignore')
from src.QCNN_layers.Pooling_layer import Pooling_3D_density
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D
import torch
import torch.nn as nn  # the neural network library of pytorch
from src import load_dataset as load
from src.training import test_net, train_net


class QCNN(nn.Module):
    def __init__(self, I, O, J, K, device):
        super(QCNN, self).__init__()
        self.conv1 = Conv_RBS_density_I2_3D(I, K, J, device)
        self.pool1 = Pooling_3D_density(I, O, J, device)
        self.conv2 = Conv_RBS_density_I2_3D(O, K, J, device)
        self.pool2 = Pooling_3D_density(O, O // 2, J, device)
        self.fc = nn.Linear((O // 2) * (O // 2) * J, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        device_cpu = torch.device("cpu")
        d = torch.stack([torch.diag(p2[i]) for i in range(batch_size)]).to(device_cpu)
        output = self.fc(d)
        return output.to(device)  # return x for visualization


# Below you can change to test
I = 16  # dimension of image we use
O = I // 2  # dimension after pooling, usually you don't need to change this
J = 2  # number of channel
k = 3  # preserving subspace parameter, usually you don't need to change this
K = 2  # size of kernel
stride = 2
batch_size = 10  # batch number
training_dataset = 10  # multiple that we reduce train dataset
testing_dataset = 10  # multiple that we reduce test dataset
is_shuffle = True
learning_rate = 2e-3
train_epochs = 2  # number of epoch we train
test_interval = 2  # when the training epoch reaches an integer multiple of the test_interval, print the testing result
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("mps")  # also torch.device("cpu"), or torch.device("mps") for macbook

train_loader, test_loader = load.load_MNIST(batch_size=batch_size, shuffle=True)
reduced_loader = load.reduce_MNIST_dataset(train_loader, training_dataset, True)
reduced_test_loader = load.reduce_MNIST_dataset(test_loader, testing_dataset, False)
network = QCNN(I, O, J, K, device)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

loss_list = []
accuracy_list = []
for epoch in range(10):
    train_loss, train_accuracy = train_net(batch_size, I, J, k, network, train_loader, criterion, optimizer,
                                           device)
    loss_list.append(train_loss)
    accuracy_list.append(train_accuracy * 100)
    print(f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %')

test_loss, test_accuracy = test_net(batch_size, I, J, k, network, reduced_test_loader, criterion, device)
print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
