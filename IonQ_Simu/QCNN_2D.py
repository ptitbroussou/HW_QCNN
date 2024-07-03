import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import torch.nn as nn
from scipy.special import binom
from src.QCNN_layers.Measurement_layer import measurement
from src.QCNN_layers.Dense_layer import Trace_out_dimension
from src.QCNN_layers.Dense_layer import Dense_RBS_density, Basis_Change_I_to_HW_density
from src.QCNN_layers.Pooling_layer import Pooling_2D_density
from src.QCNN_layers.Conv_layer import Conv_RBS_density_I2

warnings.simplefilter('ignore')

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

    def __init__(self, I, O, dense_full_gates, dense_reduce_gates, reduced_qubit, device):
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
        self.device = device
        self.conv1 = Conv_RBS_density_I2(I, 4, device)
        self.pool1 = Pooling_2D_density(I, O, device)
        # Dense layer
        self.basis_map = Basis_Change_I_to_HW_density(O, device)
        self.dense_full = Dense_RBS_density(I, dense_full_gates, device)
        self.reduce_dim = Trace_out_dimension(int(binom(reduced_qubit, 2)), device)
        self.dense_reduced = Dense_RBS_density(reduced_qubit, dense_reduce_gates, device)

    def forward(self, x):
        x = self.pool1(self.conv1(x))  # first convolution and pooling
        x = self.basis_map(x)  # basis change from 3D Image to HW=2
        x = self.dense_reduced(self.reduce_dim(self.dense_full(x)))  # dense layer
        return measurement(x, self.device)  # measure, only keep the diagonal elements
