import torch.nn.functional as F
from scipy.special import binom
from DLA import *
import time
import torch
import warnings
warnings.simplefilter('ignore')


def get_connection_cur(list_gates):
    Connections = []
    for (i,j) in list_gates:
        Connections.append(Edge(Vertex("q{}".format(i), i),Vertex("q{}".format(j), j)))
    return(Connections)

def evaluate_reduced_eff(nbr_qubits,list_gates,k, np):
    dk = int(binom(nbr_qubits,k))
    Qubits = [Vertex("q{}".format(i), i) for i in range(nbr_qubits)]
    Connections = get_connection_cur(list_gates)
    Connectivity_Graph = Network(Qubits, Connections)
    lHam_RBS = DLA_bloc_generators_RBS(k, Connectivity_Graph)
    Generators = [(torch.tensor(matrix)*1j).real.float() for matrix in lHam_RBS]
    theta_vector = [torch.tensor((torch.pi/4), requires_grad=True) for _ in range(len(list_gates))]
    # theta_vector = [torch.tensor((torch.rand(1).item() * (torch.pi * 2)), requires_grad=True) for _ in range(len(list_gates))]
    U = torch.eye(dk, dtype=torch.float).to(device)
    for index in range(len(list_gates)):
        new_U = torch.matrix_exp(Generators[index]*theta_vector[index]).to(device)
        U = new_U @ U

    U = U.real.float()
    U = U[-np:,:]
    U_vector = U.view(-1)
    Jacobian = torch.zeros(U_vector.shape[0], len(theta_vector), dtype=torch.float).to(device)

    for i in range(U_vector.shape[0]):
        for j in range(len(theta_vector)):
            grad_outputs = torch.zeros_like(U_vector, dtype=torch.float).to(device)
            grad_outputs[i] = 1.0
            grads = torch.autograd.grad(U_vector, theta_vector, grad_outputs=grad_outputs, retain_graph=True, create_graph=True, allow_unused=True)
            if grads[j] is not None:
                Jacobian[i, j] = grads[j]

    Jacobian = F.normalize(Jacobian, p=2, dim=0)
    inner_product_sum = 0.0

    for i in range(Jacobian.shape[1]):
        v_i = Jacobian[:, i]
        list_vij = []
        for j in range(Jacobian.shape[1]):
            if i!=j:
                v_j = Jacobian[:, j]
                # print("i:" + str(i) + ", j: " + str(j) + ", value: " + str(abs(torch.dot(v_i.T, v_j).item())))
                list_vij.append(abs(torch.dot(v_i.T, v_j).real))  # Use the real part of the inner product
        inner_product_sum += max(list_vij)

    return torch.linalg.matrix_rank(Jacobian.cpu()).item(), (inner_product_sum / Jacobian.shape[1]).item()

n = 5
k = 2
np = 2
device = torch.device("cpu")
initial_configuration = ([(i,j) for i in range(n) for j in range(0, n) if i!=j])
start = time.time()
rank, value = evaluate_reduced_eff(n, initial_configuration, k, np)
print("Time used: " + str((time.time()-start)) + " s")
print("Number of paramters: " + str(len(initial_configuration)))
print("Value:" + str(value) + ", rank: " + str(rank))