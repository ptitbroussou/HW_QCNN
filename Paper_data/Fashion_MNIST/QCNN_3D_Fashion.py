import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
pparent_dir_path = os.path.abspath(os.path.join(parent_dir_path, os.pardir))
sys.path.insert(0, pparent_dir_path)

import warnings
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2_3D
from src.QCNN_layers.Measurement_layer import measurement
from src.load_dataset import load_fashion_mnist
from src.QCNN_layers.Pooling_layer import Pooling_3D_density
from src.training import train_globally
from src.QCNN_layers.Dense_layer import Dense_RBS_density_3D, Basis_Change_I_to_HW_density_3D, Trace_out_dimension

warnings.simplefilter('ignore')

##################### Hyperparameters begin #######################
# Below are the hyperparameters of this network, you can change them to test
I = 16  # dimension of image we use. If you use 2 times conv and pool layers, please make it a multiple of 4
O = I // 2  # dimension after pooling, usually you don't need to change this
J = 7  # number of channel, if you use RGB dataset please let J be multiple of 3
k = 3  # preserving subspace parameter, usually you don't need to change this
K = 4  # size of kernel in the convolution layer, please make it divisible by O=I/2
stride = 2  # the difference in step sizes for different channels
batch_size = 10  # batch number
class_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # filter dataset
kernel_layout = "all_connection" # you can use "pyramid" or "all_connection"
medmnist_name = 'pathmnist'  # only useful when you use MedMNIST
train_dataset_number = int(1e3)  # training dataset sample number
test_dataset_number = int(1e3)  # testing dataset sample number
reduced_qubit = 5  # ATTENTION: please let binom(reduced_qubit,k) >= len(class_set)!
is_shuffle = True  # shuffle for this dataset
learning_rate = 1e-4  # step size for each learning steps
train_epochs = 30  # number of epoch we train
test_interval = 1  # when the training epoch reaches an integer multiple of the test_interval, print the testing result
criterion = torch.nn.CrossEntropyLoss()  # loss function
output_scale = 20
device = torch.device("cuda")  # also torch.device("cpu"), or torch.device("mps") for macbook

# Here you can modify the RBS gate list that you want for the dense layer:
# dense_full_gates is for the case qubit=O+J, dense_reduce_gates is for the case qubit=5.
# Why we need two dense gate lists? Because for the 10 labels classification we only need 10 dimension in the end,
# so after the full dense we reduce the dimension from binom(O+J,3) to binom(5,3)=10, i.e., only keep the last 5 qubits.
# Finally, we do the reduce dense for 5 qubits and measurement.
# Also, you can check visualization of different gate lists in the file "src/list_gates.py"
dense_full_gates = ([(i,j) for i in range(O+J) for j in range(0, O+J) if i>j]+
                    [(i,j) for i in range(O+J) for j in range(0, O+J) if i!=j]+
                    [(i,j) for i in range(O+J) for j in range(0, O+J) if i>j]+
                    [(i,j) for i in range(O+J) for j in range(0, O+J) if i!=j]+
                    [(i,(i+1)%(O+J)) for i in range(O+J-1)])
dense_reduce_gates = ([(i,j) for i in range(reduced_qubit) for j in range(reduced_qubit) if i>j]+
                      [(i,j) for i in range(reduced_qubit) for j in range(reduced_qubit) if i!=j]+
                      [(i,j) for i in range(reduced_qubit) for j in range(reduced_qubit) if i>j]+
                      [(i,(i+1)%(reduced_qubit)) for i in range(reduced_qubit)])
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
        self.dense_reduced1 = Dense_RBS_density_3D(0, reduced_qubit, k, dense_reduce_gates, device)

    def forward(self, x):
        x = self.pool1(self.conv1(x))  # first convolution and pooling
        x = self.pool2(self.conv2(x))  # second convolution and pooling
        x = self.basis_map(x)  # basis change from 3D Image to HW=3
        x = self.dense_reduced1(self.reduce_dim(self.dense_full1(x)))  # dense layer
        return measurement(x, device)  # measure, only keep the diagonal elements


for test in range(1):
    print("Test number: ", test)
    network = QCNN(I, O, J, K, k, kernel_layout, dense_full_gates, dense_reduce_gates, device)
    # network.load_state_dict(torch.load("FashionMNIST_modelState_75.10"))

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=1)

    # Gray MNIST/Fashion MNIST
    train_dataloader, test_dataloader = load_fashion_mnist(class_set, train_dataset_number, test_dataset_number, batch_size)
    network_state, training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list = train_globally(batch_size, I, J, network, train_dataloader, test_dataloader, optimizer, scheduler, criterion, output_scale, train_epochs, test_interval, stride, device)

    torch.save(network_state, "new_FashionMNIST_{}_modelState_75.10".format(test))  # save network parameters

    result_data = {
        'train_accuracy': training_accuracy_list,
        'train_loss': training_loss_list,
        'test_accuracy': testing_accuracy_list,
        'test_loss': testing_loss_list,
    }

    # Save the result data to a numpy file
    file_path = 'fashion_data_{}.npy'.format(test)
    np.save(file_path, result_data)
