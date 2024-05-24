import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import torch
import torch.nn as nn
from src import load_dataset as load
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D
from src.QCNN_layers.Measurement_layer import map_HW_to_measure
from src.QCNN_layers.Pooling_layer import Pooling_3D_density
from src.training import train_globally
from src.QCNN_layers.Dense_layer import Dense_RBS_density_3D, Basis_Change_I_to_HW_density_3D, Trace_out_dimension
from src.list_gates import butterfly_bi_gates, butterfly_gates, X_gates, full_connection_gates, \
    full_reverse_connection_gates, slide_gates

warnings.simplefilter('ignore')


##################### Hyperparameters begin #######################

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

# Here you can modify the RBS gate list that you want
dense_full_gates = (full_connection_gates(O + J) + butterfly_bi_gates(O + J) + butterfly_gates(O + J) +
                    full_connection_gates(O + J) + X_gates(O + J) + full_reverse_connection_gates(O + J)
                    + slide_gates(O + J))
dense_reduce_gates = (full_connection_gates(5) + butterfly_gates(5) + full_reverse_connection_gates(5) +
                      X_gates(5) + full_connection_gates(5) + full_reverse_connection_gates(5))

##################### Hyperparameters end #######################


class QCNN(nn.Module):
    """
    Pyramid Quantum convolution neural network by using Jonas's method to predict labels.
    """

    def __init__(self, I, O, J, K, k, dense_full_gates, dense_reduce_gates, device):
        """ Args:
            - I: dimension of image we use, default I is 28
            - O: dimension of image we use after a single pooling
            - J: number of convolution channels
            - K: size of kernel
            - k: preserving subspace parameter, it should be 3
            - dense_full_gates: dense gate list, dimension from binom(O+J,3) to 10
            - dense_reduce_gates: reduced dense gate list, dimension from 10 to 10
            - device: torch device (cpu, cuda, etc...)
        """
        super(QCNN, self).__init__()
        self.conv1 = Conv_RBS_density_I2_3D(I, K, J, device)
        self.pool1 = Pooling_3D_density(I, O, J, device)
        self.conv2 = Conv_RBS_density_I2_3D(O, K, J, device)
        self.pool2 = Pooling_3D_density(O, O // 2, J, device)
        self.basis_map = Basis_Change_I_to_HW_density_3D(O // 2, J, k, device)
        self.dense_full = Dense_RBS_density_3D(O // 2, J, k, dense_full_gates, device)
        self.reduce_dim = Trace_out_dimension(10, device)
        self.dense_reduced = Dense_RBS_density_3D(0, 5, k, dense_reduce_gates, device)

    def forward(self, x):
        x = self.pool1(self.conv1(x))  # first convolution and pooling
        x = self.pool2(self.conv2(x))  # second convolution and pooling
        x = self.basis_map(x)  # basis change from 3D Image to HW=3
        x = self.dense_reduced(self.reduce_dim(self.dense_full(x)))  # dense layer
        return map_HW_to_measure(x, device)  # measure, only keep the diagonal elements


network = QCNN(I, O, J, K, k, dense_full_gates, dense_reduce_gates, device)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
# Loading data
train_loader, test_loader = load.load_MNIST(batch_size=batch_size, shuffle=is_shuffle)
reduced_train_loader = load.reduce_MNIST_dataset(train_loader, training_dataset, is_train=True)
reduced_test_loader = load.reduce_MNIST_dataset(test_loader, testing_dataset, is_train=False)

# training part
network_state = train_globally(batch_size, I, J, k, network, reduced_train_loader, reduced_test_loader, optimizer, criterion, train_epochs, test_interval, stride, device)
torch.save(network_state, "model_state")
