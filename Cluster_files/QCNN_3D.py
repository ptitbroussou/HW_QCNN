import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import torch
import time
import torch.nn as nn  # the neural network library of pytorch
import load_dataset as load  # module with function to load MNIST
from Conv_Layer import Conv_RBS_density_I2_3D
from Measurement_layer import map_HW_to_measure
from Pooling_layer import Pooling_3D_density
from training import test_net, train_net, train_net_stride, test_net_stride
from Dense_layer import Dense_RBS_density_3D, Basis_Change_I_to_HW_density_3D, Trace_out_dimension
from list_gates import butterfly_bi_gates, butterfly_gates, X_gates, full_pyramid_gates, get_reduced_layers_structure
warnings.simplefilter('ignore')


class QCNN(nn.Module):
    """
    Pyramid Quantum convolution neural network by using Jonas's method to predict labels.
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

        #  Arrangement of rectangular pyramid dense gates
        list_gates_full_pyramid = get_reduced_layers_structure((O//1) + (J//1), 5)
        # Here we only keep the last 5 qubits, because 5 qubits represents 10 dimension with k=3, 10 dimension corresponds to 10 labels
        # Arrangement of pyramid dense gates with only 5 qubits
        list_gates_reduced_pyramid = full_pyramid_gates(5)
        list_gates_full = [(i, j) for i in range((O//1)+(J//1)) for j in range((O//1)+(J//1)) if i != j]
        list_gates_full_reverse = [(i, j) for i in range((O//1)+(J//1)) for j in range((O//1)+(J//1)) if i != j]
        list_gates_reduced = [(i, j) for i in range(5) for j in range(5) if i != j]
        list_gates_reduced_reverse = [(j, i) for i in range(5) for j in range(5) if i != j]
        list_gates_full_plane = [(i, i+1) for i in range((O//1) + (J//1)-1)]

        # 3 conv QCNN layers
        self.conv1 = Conv_RBS_density_I2_3D(I,K,J,device)
        self.pool1 = Pooling_3D_density(I, O, J, device)
        self.conv2 = Conv_RBS_density_I2_3D(O,K,J//1,device)
        self.pool2 = Pooling_3D_density(O, O//2, J//1, device)
        self.basis_map = Basis_Change_I_to_HW_density_3D(O//2, J//1, k, device)
        self.dense_full1 = Dense_RBS_density_3D(O//2, J//1, k, list_gates_full, device)
        self.dense_full2 = Dense_RBS_density_3D(O//2, J//1, k, butterfly_bi_gates(O//2+J), device)
        self.dense_full3 = Dense_RBS_density_3D(O//2, J//1, k, list_gates_full_reverse, device)
        self.dense_full4 = Dense_RBS_density_3D(O//2, J//1, k, butterfly_gates(O//2+J), device)
        self.dense_full5 = Dense_RBS_density_3D(O//2, J//1, k, list_gates_full, device)
        self.dense_full6 = Dense_RBS_density_3D(O//2, J//1, k, X_gates(O//2+J), device)
        self.dense_full7 = Dense_RBS_density_3D(O//2, J//1, k, list_gates_full_reverse, device)
        self.dense_full8 = Dense_RBS_density_3D(O//2, J//1, k, list_gates_full_plane, device)
        self.reduce_dim = Trace_out_dimension(10, device)
        self.dense_reduced1 = Dense_RBS_density_3D(0, 5, k, list_gates_reduced, device)
        self.dense_reduced2 = Dense_RBS_density_3D(0, 5, k, butterfly_gates(5), device)
        self.dense_reduced3 = Dense_RBS_density_3D(0, 5, k, list_gates_reduced_reverse, device)
        self.dense_reduced4 = Dense_RBS_density_3D(0, 5, k, X_gates(5), device)
        self.dense_reduced5 = Dense_RBS_density_3D(0, 5, k, list_gates_reduced, device)
        self.dense_reduced6 = Dense_RBS_density_3D(0, 5, k, list_gates_reduced_reverse, device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # c3 = self.conv3(p2)
        # p3 = self.pool3(c3)
        x = self.basis_map(x)
        x = self.dense_full7(self.dense_full7(self.dense_full6(self.dense_full5(self.dense_full4(self.dense_full3(self.dense_full2(self.dense_full1(x))))))))
        # x = self.dense_full8(self.dense_full7((self.dense_full5(self.dense_full3(self.dense_full1(x))))))
        x = self.reduce_dim(x)
        x = self.dense_reduced6(self.dense_reduced5(self.dense_reduced4(self.dense_reduced3(self.dense_reduced2(self.dense_reduced1(x))))))
        # d= (self.dense_reduced3(self.dense_reduced1(to)))
        x = map_HW_to_measure(x, device) # only keep the diagonal elements
        return x


##################### Meta-parameters begin #######################
I = 16  # dimension of image we use
O = I // 2  # dimension after pooling
J = 2  # number of channel
k = 3 # preserving subspace parameter
K = 2  # size of kernel
batch_size = 10  # batch number
scala_train = 6000  # multiple that we reduce train dataset
scala_test = 1000 # multiple that we reduce test dataset
learning_rate = 2e-3
device = torch.device("cuda") # if you are testing in your PC, you can use torch.device("cpu")
##################### Meta-parameters end #######################

# Loading data
train_loader, test_loader = load.load_MNIST(batch_size=batch_size)
reduced_loader = load.reduce_MNIST_dataset(train_loader, scala_train)
reduced_test_loader = load.reduce_MNIST_dataset(test_loader, scala_test)

conv_network = QCNN(I, O, J, K, k, device)
optimizer = torch.optim.Adam(conv_network.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# training part
loss_list = []
accuracy_list = []
for epoch in range(2):
    start = time.time()
    train_loss, train_accuracy = train_net_stride(batch_size, I, J, k, conv_network, train_loader, criterion, optimizer, device)
    loss_list.append(train_loss)
    accuracy_list.append(train_accuracy * 100)
    end = time.time()
    print(f'Epoch {epoch}: Loss = {train_loss:.6f}, accuracy = {train_accuracy * 100:.4f} %, time={(end-start):.4f}s')
    if ((epoch % 5 == 0)):
        # if ((epoch % 5 == 0) and (epoch != 0)):
        test_loss, test_accuracy = test_net_stride(batch_size, I, J, k, conv_network, reduced_test_loader, criterion, device)
        print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')

# testing part
test_loss, test_accuracy = test_net_stride(batch_size, I, J, k, conv_network, reduced_test_loader, criterion, device)
print(f'Evaluation on test set: Loss = {test_loss:.6f}, accuracy = {test_accuracy * 100:.4f} %')
#%%
