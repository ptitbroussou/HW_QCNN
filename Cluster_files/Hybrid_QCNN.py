import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.simplefilter('ignore')
from Pooling_layer import Pooling_3D_density
from Conv_Layer import Conv_RBS_density_I2_3D
import torch
import torch.nn as nn  # the neural network library of pytorch
import load_dataset as load  # module with function to load MNIST
from training import test_net, train_net


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


I = 12  # dimension of image we use
O = I // 2  # dimension after pooling
J = 2  # number of channel
K = 2  # size of kernel
k = 3
batch_size = 10  # batch number
scala = 6000  # time we reduce dataset
learning_rate = 1e-2
device = torch.device("cuda")

train_loader, test_loader = load.load_MNIST(batch_size=batch_size)
reduced_loader = load.reduce_MNIST_dataset(train_loader, scala)
reduced_test_loader = load.reduce_MNIST_dataset(test_loader, 100)

conv_network = QCNN(I, O, J, K, device)
optimizer = torch.optim.Adam(conv_network.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

loss_list = []
accuracy_list = []
for epoch in range(10):
    train_loss, train_accuracy = train_net(batch_size, I, J, k, conv_network, train_loader, criterion, optimizer,
                                           device)
    loss_list.append(train_loss)
    accuracy_list.append(train_accuracy * 100)
    print(f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %')

test_loss, test_accuracy = test_net(batch_size, I, J, k, conv_network, reduced_test_loader, criterion, device)
print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
