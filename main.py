import torch
from torch import nn
from scipy.special import binom

from RBS_Circuit import *
from Conv_Layer import *
from Dense import *
#from PennyLane_toolbox import *
from toolbox import RBS_generalized, map_RBS
from Pooling import *

# Hardware accelerator:
device = torch.device("cpu")  # Only using CPU

n, k = 16, 2

### Initial state in the Image basis:
initial_state_1 = torch.tensor([1.0 for i in range((n//2)**2)])
initial_state_1 = initial_state_1/torch.norm(initial_state_1)
initial_state_2 = torch.zeros(((n//2)**2))
initial_state_2[0] = 1.0
# Reshaping the initial states
ini_1 = initial_state_1.unsqueeze(0)
ini_2 = initial_state_2.unsqueeze(0)
# We now consider a batch of Image:
batch = torch.stack((initial_state_1, initial_state_2))



# Define the Pytorch model of the QCNN architecture:
gate_dense = [(0,1),(1,2),(2,3),(0,1),(1,2),(2,3)]

model = nn.Sequential(Conv_RBS_state_vector_I2(8,4,device), Pooling_2D_state_vector(8, 4,device), Conv_RBS_state_vector_I2(4,2,device), Pooling_2D_state_vector(4, 2,device), Basis_Change_I_to_HW(2, device), Dense_RBS_state_vector(2,gate_dense,device))


print(ini_1.size(), ini_2.size(), batch.size())

out_1 = model(ini_1)
out_2 = model(ini_2)
out_batch = model(batch)

print(out_1.size(), out_2.size(), out_batch.size())