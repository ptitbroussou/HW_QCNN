import torch
from torch import nn
from scipy.special import binom

from RBS_Circuit import *
from toolbox import RBS_generalized, map_RBS

# Hardware accelerator:
device = torch.device("cpu")  # Only using CPU

n, k = 4, 2
i, j = 1, 2

initial_state = torch.zeros(int(binom(n,k)))
initial_state[0] = 1
rho = torch.outer(initial_state, initial_state.t())

rho_list = [[0 for i in range(int(binom(n,k)))] for j in range(int(binom(n,k)))]
rho_list[0][0] = 1
batch_ini = torch.tensor([[1,0,0,0,0,0] for i in range(5)],dtype=torch.float)
batch_rho = torch.tensor([rho_list for i in range(5)],dtype=torch.float)


angle = torch.nn.Parameter(torch.randn(1)) 
VQC_state = RBS_VQC_state_vector(n, k, [(i,j)], device)
VQC_density = RBS_VQC_density(n, k, [(i,j)], device)
VQC_state.RBS_gates[0].angle = angle
VQC_density.RBS_gates[0].angle = angle


a_0 = VQC_state(initial_state)
a_1 = VQC_state(batch_ini)
b_0 = VQC_density(rho)
b_1 = VQC_density(batch_rho)

