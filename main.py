import torch
from torch import nn
from scipy.special import binom

from RBS_Circuit import *
from Conv_Layer import *
#from PennyLane_toolbox import *
from toolbox import RBS_generalized, map_RBS
from Pooling import *

# Hardware accelerator:
device = torch.device("cpu")  # Only using CPU

n, k = 8, 2
i, j = 0, 1

initial_state = torch.zeros(int(binom(n,k)))
initial_state[3] = 1
rho = torch.outer(initial_state, initial_state.t())

nbr_batch = 5
rho_list = [[0 for i in range(int(binom(n,k)))] for j in range(int(binom(n,k)))]
rho_list[3][3] = 1
ini_state_list = [0 for i in range(int(binom(n,k)))]
ini_state_list[3] = 1
batch_ini = torch.tensor([ini_state_list for i in range(nbr_batch)],dtype=torch.float)
batch_rho = torch.tensor([rho_list for i in range(nbr_batch)],dtype=torch.float)


angle = torch.nn.Parameter(torch.randn(1)) 
VQC_state = RBS_VQC_state_vector(n, k, [(i,j)], device)
VQC_density = RBS_VQC_density(n, k, [(i,j)], device)
VQC_state.RBS_gates[0].angle = angle
VQC_density.RBS_gates[0].angle = angle


a_0 = VQC_state(initial_state)
a_1 = VQC_state(batch_ini)
b_0 = VQC_density(rho)
b_1 = VQC_density(batch_rho)

CONV_state = Conv_RBS_state_vector(4, 2, device)
CONV_density = Conv_RBS_density(4,2, device)

c_0 = CONV_state(initial_state)
c_1 = CONV_state(batch_ini)
d_0 = CONV_density(rho)
d_1 = CONV_density(batch_rho)

ini = torch.tensor([1.0 for i in range(4**2)])
ini = ini/torch.linalg.norm(ini)

POOL_state = Pooling_2D_state_vector(4, 2, device)

e_0 = POOL_state(ini)