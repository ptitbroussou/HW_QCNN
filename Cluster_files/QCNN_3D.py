import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.simplefilter('ignore')
from Pooling import Pooling_2D_density_3D
from Conv_Layer import Conv_RBS_density_I2_3D
import torch
import torch.nn as nn  # the neural network library of pytorch
import load_dataset_letao as load  # module with function to load MNIST
from toolbox import reduce_MNIST_dataset
from training import test_net, train_net
from Dense import Dense_RBS_density_3D
from toolbox import Basis_Change_I_to_HW_density_3D, Trace_out_dim, get_reduced_layers_structure, PQNN_building_brick, map_HW_to_measure


class QCNN(nn.Module):
    def __init__(self, I, O, J, K, k, device):
        super(QCNN, self).__init__()
        list_gates_pyramid = get_reduced_layers_structure(O+J, 5)
        list_gates_pyramid_small = []
        PQNN_param_dictionary, PQNN_dictionary, PQNN_layer = PQNN_building_brick(0, 5, index_first_RBS=0, index_first_param=0)
        for x, y in PQNN_dictionary.items():
            list_gates_pyramid_small.append((y,y+1))

        self.conv1 = Conv_RBS_density_I2_3D(I,K,J,device)
        self.pool1 = Pooling_2D_density_3D(I, O, J, device)
        self.conv2 = Conv_RBS_density_I2_3D(O,K,J,device)
        self.pool2 = Pooling_2D_density_3D(O, O//2, J, device)
        self.basis_map = Basis_Change_I_to_HW_density_3D(O//2, J, k, device)
        self.dense1 = Dense_RBS_density_3D(O//2, J, k, list_gates_pyramid, device)
        self.tomo = Trace_out_dim(10, device)
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
        output = map_HW_to_measure(d2, device)
        return output    # return x for visualization


I = 12 # dimension of image we use
O = I//2 # dimension after pooling
J = 2 # number of channel
k = 3
K = 2 # size of kernel
batch_size = 10 # batch number
scala = 6000 # time we reduce dataset
learning_rate = 1e-1
device = torch.device("cuda")

train_loader, test_loader, dim_in, dim_out = load.load_MNIST(batch_size=batch_size)
reduced_loader = reduce_MNIST_dataset(train_loader, scala)
reduced_test_loader = reduce_MNIST_dataset(test_loader, 100)

conv_network = QCNN(I, O, J, K, k, device)
optimizer = torch.optim.Adam(conv_network.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

loss_list = []
accuracy_list = []
for epoch in range(10):
    train_loss, train_accuracy = train_net(batch_size, I, J, k, conv_network, train_loader, criterion, optimizer, device)
    loss_list.append(train_loss)
    accuracy_list.append(train_accuracy*100)
    print(f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy*100:.4f} %')


test_loss, test_accuracy = test_net(batch_size, I, J, k, conv_network, reduced_test_loader, criterion, device)
print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy*100:.4f} %')