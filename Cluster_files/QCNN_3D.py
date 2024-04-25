import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

warnings.simplefilter('ignore')
from Pooling import Pooling_2D_density_3D
from Conv_Layer import Conv_RBS_density_I2_3D
import torch
import torch.nn as nn  # the neural network library of pytorch
import load_dataset_letao as load  # module with function to load MNIST
from toolbox import reduce_MNIST_dataset, get_full_pyramid_gates
from training import test_net, train_net
from Dense import Dense_RBS_density_3D
from toolbox import Basis_Change_I_to_HW_density_3D, Trace_out_dim, get_reduced_layers_structure, PQNN_building_brick, \
    map_HW_to_measure


class QCNN(nn.Module):
    """
    Pyramid Quantum convolution neural network by using Jonas's method to predict labels.
    You can check the figure in https://www.mathcha.io/editor/NOJPecYwiOphNYrDX9fG2eOznc07MJ1gfLL4Olo
    """
    def __init__(self, I, O, J, K, k, device):
        """ Args:
            - I: dimension of image we use, default I is 28
            - O: dimension of image we use after a single pooling
            - J: number of convolution channels
            - K: size of kernel
            - k: preserving subspace parameter, it should be 3
            - device: torch device (cpu, cuda, etc...)
        """
        super(QCNN, self).__init__()

        # Arrangement of rectangular pyramid dense gates
        list_gates_pyramid = get_reduced_layers_structure(O + J, 5)
        # Here we only keep the last 5 qubits, because 5 qubits represents 10 dimension with k=3, 10 dimension corresponds to 10 labels
        # Arrangement of pyramid dense gates with only 5 qubits
        list_gates_pyramid_small = get_full_pyramid_gates(5)

        # QCNN layers
        self.conv1 = Conv_RBS_density_I2_3D(I, K, J, device)
        self.pool1 = Pooling_2D_density_3D(I, O, J, device)
        self.conv2 = Conv_RBS_density_I2_3D(O, K, J, device)
        self.pool2 = Pooling_2D_density_3D(O, O // 2, J, device)
        self.basis_map = Basis_Change_I_to_HW_density_3D(O // 2, J, k, device)
        self.dense1 = Dense_RBS_density_3D(O // 2, J, k, list_gates_pyramid, device)
        self.tomo = Trace_out_dim(10, device) # only keep the last 10 dimension elements
        self.dense2 = Dense_RBS_density_3D(0, 5, k, list_gates_pyramid_small, device)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        b1 = self.basis_map(p2)
        d1 = self.dense1(b1)
        to = self.tomo(d1)
        d2 = self.dense2(to)
        output = map_HW_to_measure(d2, device) # only keep the diagonal elements
        return output


##################### Meta-parameters begin #######################
I = 12  # dimension of image we use
O = I // 2  # dimension after pooling
J = 2  # number of channel
k = 3 # preserving subspace parameter
K = 2  # size of kernel
batch_size = 10  # batch number
scala_train = 6000  # multiple that we reduce train dataset
scala_test = 100 # multiple that we reduce test dataset
learning_rate = 1e-1
device = torch.device("cuda") # if you are testing in your PC, you can use torch.device("cpu")
##################### Meta-parameters end #######################

# Loading data
train_loader, test_loader, dim_in, dim_out = load.load_MNIST(batch_size=batch_size)
reduced_loader = reduce_MNIST_dataset(train_loader, scala_train)
reduced_test_loader = reduce_MNIST_dataset(test_loader, scala_test)

conv_network = QCNN(I, O, J, K, k, device)
optimizer = torch.optim.Adam(conv_network.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# training part
loss_list = []
accuracy_list = []
for epoch in range(10):
    train_loss, train_accuracy = train_net(batch_size, I, J, k, conv_network, train_loader, criterion, optimizer, device)
    loss_list.append(train_loss)
    accuracy_list.append(train_accuracy * 100)
    print(f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %')

# testing part
test_loss, test_accuracy = test_net(batch_size, I, J, k, conv_network, reduced_test_loader, criterion, device)
print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
