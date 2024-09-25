import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from scipy.special import binom
from src.QCNN_layers.Measurement_layer import measurement
from src.load_dataset import load_fashion_mnist, load_mnist
from src.training import train_globally_2D
from src.QCNN_layers.Dense_layer import Trace_out_dimension
from src.list_gates import drip_circuit, full_pyramid_circuit, half_connection_circuit, full_connection_circuit, slide_circuit
from src.QCNN_layers.Dense_layer import Dense_RBS_density, Basis_Change_I_to_HW_density
from src.QCNN_layers.Pooling_layer import Pooling_2D_density
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2
warnings.simplefilter('ignore')

##################### Hyperparameters begin #######################
# Below are the hyperparameters of this network, you can change them to test
I = 8  # dimension of image we use. If you use 2 times conv and pool layers, please make it a multiple of 4
O = I // 2  # dimension after pooling, usually you don't need to change this
k = 2  # preserving subspace parameter, usually you don't need to change this
K = 4  # size of kernel in the convolution layer, please make it divisible by O=I/2
batch_size = 10  # batch number
class_set = [i for i in range(6)]  # filter dataset, here I set 6-labels
train_dataset_number = 100  # training dataset sample number
test_dataset_number = 100  # testing dataset sample number
reduced_qubit = 4  # ATTENTION: let binom(reduced_qubit,k) >= len(class_set)!
is_shuffle = False  # shuffle for this dataset
learning_rate = 1e-2  # step size for each learning steps
train_epochs = 20  # number of epoch we train
test_interval = 10  # when the training epoch reaches an integer multiple of the test_interval, print the testing result
output_scale = 30
criterion = torch.nn.CrossEntropyLoss()  # loss function
device = torch.device("cpu")  # also torch.device("cpu"), or torch.device("mps") for macbook

# Here you can modify the RBS gate list that you want for the dense layer:
# dense_full_gates is for the case qubit=O+J, dense_reduce_gates is for the case qubit=5.
# Why we need two dense gate lists? Because for the 10 labels classification we only need 10 dimension in the end,
# so after the full dense we reduce the dimension from binom(O+J,3) to binom(5,3)=10, i.e., only keep the last 5 qubits.
# Finally, we do the reduce dense for 5 qubits and measurement.
# Also, you can check visualization of different gate lists in the file "src/list_gates.py"

dense_full_gates = half_connection_circuit(O) + full_connection_circuit(O) + half_connection_circuit(
    O) + full_connection_circuit(O) + slide_circuit(O)
dense_reduce_gates = half_connection_circuit(reduced_qubit) + full_connection_circuit(
    reduced_qubit) + half_connection_circuit(reduced_qubit) + slide_circuit(reduced_qubit)

# other simpler dense layer layout
# dense_full_gates = drip_circuit(O)
# dense_reduce_gates = full_pyramid_circuit(reduced_qubit)

##################### Hyperparameters end #######################

class QCNN(nn.Module):
    """
    Hamming weight preserving quantum convolution neural network (k=2)
    """

    def __init__(self, I, O, dense_full_gates, dense_reduce_gates, device):
        """ Args:
            - I: dimension of image we use, default I is 28
            - O: dimension of image we use after a single pooling
            - dense_full_gates: dense gate list, dimension from binom(O,2) to binom(4,2)=6
            - dense_reduce_gates: reduced dense gate list, dimension from 10 to 10
            - device: torch device (cpu, cuda, etc...)
        """
        super(QCNN, self).__init__()
        self.conv1 = Conv_RBS_density_I2(I, K, device)
        self.pool1 = Pooling_2D_density(I, O, device)
        self.conv2 = Conv_RBS_density_I2(O, K, device)
        self.pool2 = Pooling_2D_density(O, O // 2, device)
        self.basis_map = Basis_Change_I_to_HW_density(O // 2, device)
        self.dense_full = Dense_RBS_density(O, dense_full_gates, device)
        self.reduce_dim = Trace_out_dimension(int(binom(reduced_qubit, k)), device)
        self.dense_reduced = Dense_RBS_density(reduced_qubit, dense_reduce_gates, device)

    def forward(self, x):
        x = self.pool1(self.conv1(x))  # first convolution and pooling
        x = self.pool2(self.conv2(x))  # second convolution and pooling
        x = self.basis_map(x)  # basis change from 2D Image to HW=2
        x = self.dense_reduced(self.reduce_dim(self.dense_full(x)))  # dense layer
        return measurement(x, device)  # measure, only keep the diagonal elements


network = QCNN(I, O, dense_full_gates, dense_reduce_gates, device)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Loading data
print("Loading dataset...")
train_dataloader, test_dataloader = load_fashion_mnist(class_set, train_dataset_number, test_dataset_number, batch_size)
# train_dataloader, test_dataloader = load_mnist(class_set, train_dataset_number, test_dataset_number, batch_size)

# training part
network_state, training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list = train_globally_2D(batch_size, I, network, train_dataloader, test_dataloader, optimizer, scheduler,
                                  criterion, output_scale, train_epochs, test_interval, device)
# Saving network parameters
torch.save(network_state, "Model_states/QCNN_2DmodelState")  # save network parameters
result_data = {'train_accuracy': training_accuracy_list,'train_loss': training_loss_list,'test_accuracy': testing_accuracy_list,'test_loss': testing_loss_list,}
file_path = 'Model_states/plot_data_2D.npy'
np.save(file_path, result_data)