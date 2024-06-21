import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from scipy.special import binom
from src.QCNN_layers.Measurement_layer import measurement
from src.load_dataset import load_fashion_mnist, load_mnist
from src.training import train_globally_2D
from src.QCNN_layers.Dense_layer import Trace_out_dimension
from src.list_gates import drip_circuit, full_pyramid_circuit
from src.QCNN_layers.Dense_layer import Dense_RBS_density, Basis_Change_I_to_HW_density
from src.QCNN_layers.Pooling_layer import Pooling_2D_density
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2

warnings.simplefilter('ignore')


##################### Hyperparameters begin #######################
# Below are the hyperparameters of this network, you can change them to test
I = 8  # dimension of image we use. If you use 2 times conv and pool layers, please make it a multiple of 4
O = I // 2  # dimension after pooling, usually you don't need to change this
k = 2  # preserving subspace parameter, usually you don't need to change this
K = 2  # size of kernel in the convolution layer, please make it divisible by O=I/2
batch_size = 10  # batch number
class_set = [0, 1]  # filter dataset
train_dataset_number = 1e4  # training dataset sample number
test_dataset_number = 1e4  # testing dataset sample numbers
reduced_qubit = 3  # ATTENTION: let binom(reduced_qubit,k) >= len(class_set)!
is_shuffle = False  # shuffle for this dataset
learning_rate = 1e-1  # step size for each learning steps
train_epochs = 10  # number of epoch we train
test_interval = 10  # when the training epoch reaches an integer multiple of the test_interval, print the testing result
criterion = torch.nn.CrossEntropyLoss()  # loss function
device = torch.device("cuda")  # also torch.device("cpu"), or torch.device("mps") for macbook

# Here you can modify the RBS gate list that you want for the dense layer:
# dense_full_gates is for the case qubit=O+J, dense_reduce_gates is for the case qubit=5.
# Why we need two dense gate lists? Because for the 10 labels classification we only need 10 dimension in the end,
# so after the full dense we reduce the dimension from binom(O+J,3) to binom(5,3)=10, i.e., only keep the last 5 qubits.
# Finally, we do the reduce dense for 5 qubits and measurement.
# Also, you can check visualization of different gate lists in the file "src/list_gates.py"
dense_full_gates = drip_circuit(I)
dense_reduce_gates = full_pyramid_circuit(reduced_qubit)


##################### Hyperparameters end #######################

class QCNN(nn.Module):
    """
    Hamming weight preserving quantum convolution neural network (k=3)

    Tensor dataflow of this network:
    input density matrix: (batch,J*I^2,J*I^2)--> conv1: (batch,J*I^2,J*I^2)--> pool1: (batch,J*O^2,J*O^2)
    --> conv2: (batch,J*O^2,J*O^2)--> pool2: (batch,J*(O/2)^2,J*(O/2)^2)--> basis_map: (batch,binom(O+J,3),binom(O+J,3))
    --> full_dense: (batch,binom(O+J,3),binom(O+J,3)) --> reduce_dim: (batch,binom(5,3)=10,10)
    --> reduce_dense: (batch,10,10) --> output measurement: (batch,10)

    Then we can use it to calculate the Loss(output, targets)
    """

    def __init__(self, I, O, dense_full_gates, dense_reduce_gates, device):
        """ Args:
            - I: dimension of image we use, default I is 28
            - O: dimension of image we use after a single pooling
            - J: number of convolution channels
            - K: size of kernel
            - k: preserving subspace parameter, it should be 3
            - dense_full_gates: dense gate list, dimension from binom(O+J,3) to binom(5,3)=10
            - dense_reduce_gates: reduced dense gate list, dimension from 10 to 10
            - device: torch device (cpu, cuda, etc...)
        """
        super(QCNN, self).__init__()
        self.conv1 = Conv_RBS_density_I2(I, 4, device)
        self.pool1 = Pooling_2D_density(I, O, device)
        # Dense layer
        self.basis_map = Basis_Change_I_to_HW_density(O, device)
        self.dense_full = Dense_RBS_density(I, dense_full_gates, device)
        self.reduce_dim = Trace_out_dimension(int(binom(reduced_qubit, k)), device)
        self.dense_reduced = Dense_RBS_density(reduced_qubit, dense_reduce_gates, device)

    def forward(self, x):
        x = self.pool1(self.conv1(x))  # first convolution and pooling
        x = self.basis_map(x)  # basis change from 3D Image to HW=2
        x = self.dense_reduced(self.reduce_dim(self.dense_full(x)))  # dense layer
        return measurement(x, device)  # measure, only keep the diagonal elements


network = QCNN(I, O, dense_full_gates, dense_reduce_gates, device)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Loading data
train_dataloader, test_dataloader = load_fashion_mnist(class_set, train_dataset_number, test_dataset_number, batch_size)
# train_dataloader, test_dataloader = load_mnist(class_set, train_dataset_number, test_dataset_number, batch_size)

# training part
network_state = train_globally_2D(batch_size, I, network, train_dataloader, test_dataloader, optimizer, scheduler,
                                  criterion, train_epochs, test_interval, device)