"""
===================================================================================
This file can be executed directly, it shows the basic structure of our HW3-QCNN,
and you can also do further testing by adjusting the hyperparameter, enjoy :)
===================================================================================
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D
from src.QCNN_layers.Measurement_layer import measurement
from src.load_dataset import load_mnist, load_fashion_mnist, load_cifar10
from src.QCNN_layers.Pooling_layer import Pooling_3D_density
from src.training import train_globally
from src.list_gates import slide_circuit, full_connection_circuit, half_connection_circuit
from src.QCNN_layers.Dense_layer import Dense_RBS_density_3D, Basis_Change_I_to_HW_density_3D, Trace_out_dimension

warnings.simplefilter('ignore')

########################## Hyperparameters BEGIN ############################
# Below are the hyperparameters of this network, you may change them to test
I = 8  # dimension of image we use. If you use 2 times conv and pool layers, please make it a multiple of 4
J = 4  # number of channel for convolution
K = 4  # size of kernel in the convolution layer, please make it divisible by O=I/2
stride = 2  # the difference in step sizes for different channels
batch_size = 10  # batch number
kernel_layout = "all_connection"  # you can use "pyramid" or "all_connection"
train_dataset_number = 20  # training dataset sample number
test_dataset_number = 20  # testing dataset sample number
learning_rate = 1e-2 * 0.66  # step size for each learning steps
gamma = 0.9  # multiplicative factor of learning rate decay
train_epochs = 10  # number of epoch we train
test_interval = 10  # when the training epoch reaches an integer multiple of the test_interval, print the testing result
output_scale = 30  # Recommended range [10,50]. depends on your other parameters
device = torch.device("cpu")  # also torch.device("cuda"), or torch.device("mps") for macbook

# Below are the other hyperparameters of this network, usually you don't need to change this
O = I // 2  # dimension of image data after one pooling
k = 3  # preserving subspace parameter, k=3 for multichannel images, k=2 for single channel images
class_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # filter dataset, 10 labels by default
reduced_qubit = 5  # ATTENTION: please let binom(reduced_qubit,k) >= len(class_set)! 5 qubits for 10 labels by default
is_shuffle = True  # shuffle for this dataset
criterion = torch.nn.CrossEntropyLoss()  # loss function

# Here you can modify the dense layer layout, i.e., RBS gate list:
# dense_full_gates is for the case qubit=O+J, dense_reduce_gates is for the case qubit=5.
# Also, you can check visualization of different gate lists in the file "src/list_gates.py"
dense_full_gates = half_connection_circuit(O + J) + full_connection_circuit(O + J) + half_connection_circuit(
    O + J) + full_connection_circuit(O + J) + slide_circuit(O + J - 1)
dense_reduce_gates = half_connection_circuit(reduced_qubit) + full_connection_circuit(
    reduced_qubit) + half_connection_circuit(reduced_qubit) + slide_circuit(reduced_qubit)


########################## Hyperparameters END ############################


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

    def __init__(self, I, O, J, K, k, kernel_layout, dense_full_gates, dense_reduce_gates, device):
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
        self.conv1 = Conv_RBS_density_I2_3D(I, K, J, kernel_layout, device)
        self.pool1 = Pooling_3D_density(I, O, J, device)
        self.conv2 = Conv_RBS_density_I2_3D(O, K, J, kernel_layout, device)
        self.pool2 = Pooling_3D_density(O, O // 2, J, device)
        self.basis_map = Basis_Change_I_to_HW_density_3D(O // 2, J, k, device)
        self.dense_full1 = Dense_RBS_density_3D(O // 2, J, k, dense_full_gates, device)
        self.reduce_dim = Trace_out_dimension(len(class_set), device)
        self.dense_reduced = Dense_RBS_density_3D(0, reduced_qubit, k, dense_reduce_gates, device)

    def forward(self, x):
        x = self.pool1(self.conv1(x))  # first convolution and pooling
        x = self.pool2(self.conv2(x))  # second convolution and pooling
        x = self.basis_map(x)  # basis change from 3D Image to HW=3
        x = self.dense_reduced(self.reduce_dim(self.dense_full1(x)))  # dense layer
        return measurement(x, device)  # measure, only keep the diagonal elements


network = QCNN(I, O, J, K, k, kernel_layout, dense_full_gates, dense_reduce_gates, device)
# network.load_state_dict(torch.load("QCNN_modelState")) # you can load the network parameter file, otherwise it will be initialized randomly

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=gamma) # learning rate decay

# Loading dataset, you can choose the dataset you want to use: MNIST/FashionMNIST/CIFAR-10
train_dataloader, test_dataloader = load_mnist(class_set, train_dataset_number, test_dataset_number, batch_size)
# train_dataloader, test_dataloader = load_cifar10(class_set, train_dataset_number, test_dataset_number, batch_size)
# train_dataloader, test_dataloader = load_fashion_mnist(class_set, train_dataset_number, test_dataset_number, batch_size)

# Starting training
network_state, _, _, _, _ = train_globally(batch_size, I, J, network, train_dataloader, test_dataloader, optimizer, scheduler, criterion, output_scale,train_epochs, test_interval, stride, device)
# Saving network parameters
torch.save(network_state, "Model_states/QCNN_modelState")
