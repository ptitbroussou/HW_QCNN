import torch
from torch import nn
from scipy.special import binom

from RBS_Circuit import *
from Conv_Layer import *
from Dense import *
#from PennyLane_toolbox import *
from toolbox import RBS_generalized, map_RBS
from Pooling import *
from dataset import *
from training import *

# Hardware accelerator:
device = torch.device("cpu")  # Only using CPU

n, k = 8, 2

### Initial state in the Image basis:
initial_state_1 = torch.tensor([1.0 for i in range((n//2)**2)])
initial_state_1 = initial_state_1/torch.norm(initial_state_1)
initial_state_2 = torch.zeros(((n//2)**2))
initial_state_2[0] = 1.0
# Reshaping the initial states
ini_1 = initial_state_1.unsqueeze(0)
rho_1 = torch.einsum('bi, bo->bio', ini_1, ini_1)
ini_2 = initial_state_2.unsqueeze(0)
rho_2 = torch.einsum('bi, bo->bio', ini_2, ini_2)
# We now consider a batch of Image:
batch_state = torch.stack((initial_state_1, initial_state_2))
batch_density = torch.stack((rho_1, rho_2))


# Define the Pytorch model of the QCNN architecture:
gate_dense = [(0,1),(1,2),(2,3),(0,1),(1,2),(2,3)]
model_state = nn.Sequential(Conv_RBS_density_I2(4,2,device), Pooling_2D_density(4,2,device), Basis_Change_I_to_HW_density(2, device), Dense_RBS_density(2, gate_dense, device))

# Data Loading
I, batch_size, nbr_class, nbr_train, nbr_test = 28, 1, 1, 500, 100
mnistTrainLoader, mnistTestLoader = MNIST_DataLoading(I, batch_size, nbr_class, nbr_train, nbr_test)

#model = QCNN_model(model_state, nbr_epochs=5, loss_fn=nn.MSELoss, device=device)
#X_train, y_train, X_test, y_test, _ =  Fashion_MNIST_pca([i for i in range(2)], 500, 100, Npca=8**2, I=8, O=int(binom(4,2)))
