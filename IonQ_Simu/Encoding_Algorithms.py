import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.special import binom
from tqdm import tqdm

from IonQ_Simu.QFIM import QFIM_RBS_subspace
from IonQ_Simu.toolbox import RBS_generalized, map_RBS
from src.toolbox import map_RBS_Image_HW2, dictionary_RBS_I2_2D


################################################################################
# Designing the quantum dataloader:
################################################################################
def power_parallelization(edge, list_gates):
    """ This function output the maximum number of gates in the circuit descibed
    by list_gates that edge could be parallelize with. """
    i, parallelization_possible = 0, True
    parallelizability  = 0
    while (i<len(list_gates)) and parallelization_possible:
        reverse_index = len(list_gates) - (i+1)
        if (edge.tuple()[0] not in list_gates[reverse_index]) and (edge.tuple()[1] not in list_gates[reverse_index]):
            parallelizability += 1
        else:
            parallelization_possible = False
        i += 1
    return(parallelizability)

def sorted_to_minimize_depth(list_gates, ListEdges):
    """ This function output a list of edges sorted in a way such that the first
    gates in this list are more likely to be parallelize with the previous gates
    in list_gates and thus minimize the depth of the circuit. """
    sorted_list_Edges = []
    List_Edges_copy = ListEdges.copy()
    for nbr_edge in range(len(ListEdges)):
        best_edge, index_best_edge, max_parallelizability = List_Edges_copy[0], 0, power_parallelization(List_Edges_copy[0], list_gates)
        for index, edge in enumerate(List_Edges_copy[1:]):
            parallelizability = power_parallelization(edge, list_gates)
            if (parallelizability > max_parallelizability):
                best_edge, index_best_edge, max_parallelizability = edge, index+1, parallelizability
        sorted_list_Edges.append(best_edge)
        List_Edges_copy.pop(index_best_edge)
    return(sorted_list_Edges)

def RBS_tensor_encoding(I, Connectivity_Graph, index_input_state=0):
    """ This function is similar to RBS_subspace_encoding_1 but test in priority
    the gates that are more likely to reduce the circuit depth. I is the dimension
    of the input state. """
    max_rank, list_gates = I**2 - 1, []
    pbar = tqdm(total = max_rank)
    RBS_dictionnary, mapping_RBS = {}, map_RBS(2*I,2) # Link between states and egdes
    for edge in Connectivity_Graph.ListEdges:
        RBS_dictionnary[edge] = RBS_generalized(edge.tuple()[0], edge.tuple()[1], 2*I, 2, mapping_RBS)
    # Initialization: the first RBS must impact the input_state
    index_edge, initial_state_flag = 0,  False
    while (not initial_state_flag) and (index_edge < len(Connectivity_Graph.ListEdges)):
        edge = Connectivity_Graph.ListEdges[index_edge]
        for couple_states in RBS_dictionnary[edge]:
            if (not initial_state_flag and (index_input_state in couple_states)):
                list_gates.append(edge.tuple())
                initial_state_flag = True
                pbar.update(1)
        index_edge += 1
    if not initial_state_flag:
        print("Impossible to design a Quantum Data Loader")
        return(list_gates)
    # Other gates chosen according to the rank of the QFIM
    input_state = np.zeros(int(binom(2*I,2)))
    input_state[index_input_state] = 1
    theta = 2*np.pi*np.random.rand(len(list_gates))
    QFIM = QFIM_RBS_subspace(list_gates, input_state, 2*I, 2, theta)
    index_RBS, rank_QFIM = 0, np.linalg.matrix_rank(QFIM)
    #print("Rank QFIM = {}".format(rank_QFIM))
    # We sort the edges in order to minimize the circuit depth
    ListEdges_sorted = sorted_to_minimize_depth(list_gates, Connectivity_Graph.ListEdges)
    while (index_RBS < len(RBS_dictionnary.keys())) and (rank_QFIM < max_rank):
        edge = ListEdges_sorted[index_RBS]
        list_gates.append(edge.tuple())
        theta = 2*np.pi*np.random.rand(len(list_gates))
        QFIM = QFIM_RBS_subspace(list_gates, input_state, 2*I, 2, theta)
        if (np.linalg.matrix_rank(QFIM) > rank_QFIM):
            rank_QFIM += 1
            #print("Rank QFIM = {}".format(rank_QFIM))
            # We sort the edges in order to minimize the circuit depth
            ListEdges_sorted = sorted_to_minimize_depth(list_gates, Connectivity_Graph.ListEdges)
            index_RBS = 0
            pbar.update(1)
        else:
            list_gates.pop()
            index_RBS += 1
    if (rank_QFIM == max_rank):
        print("Quantum Data Loader successfully designed")
        pbar.close()
    else:
        print("Impossible to design a Quantum Data Loader")
        pbar.close()
    return(list_gates)


################################################################################
# Training the quantum dataloader:
################################################################################
def global_training_quantum_data_loader(QDL_module, train_init_states, test_init_states, device):
    """ This function trains the Quantum Data Loader module and outputs the set of
    parameters corresponding to the tensor encoding of each initial training and testing
    states.
    Args:
        - QDL_module: Quantum Data Loader module (from src.RBS_VQCs.py)
        - train_init_states: tensor of shape (train_dataset_number, I**2)
        - test_init_states: tensor of shape (test_dataset_number, I**2)
    Output:
        - train_encoded_parameters:
        - test_encoded_parameters:
    """
    num_parameters = len(QDL_module.RBS_gates)
    I = int(np.sqrt(train_init_states.shape[1]))
    dict_I2, map_RBS_HW2 = dictionary_RBS_I2_2D(I), map_RBS(2*I,2)
    train_encoded_parameters, test_encoded_parameters = torch.zeros(train_init_states.shape[0], num_parameters), torch.zeros(test_init_states.shape[0], num_parameters)
    # Initial state in the circuit:
    initial_state = torch.zeros(int(binom(2*I,2)), device=device)
    initial_state[0] = 1
    # Training parameters:
    criterion = torch.nn.MSELoss()  # loss function
    optimizer = torch.optim.Adam(QDL_module.parameters(), lr=0.1)
    nbr_epoch = int(1e2)
    # Training samples:
    print("Encoding training samples:")
    for i in tqdm(range(train_init_states.shape[0])):
        sample_basis_HW2 = torch.from_numpy(map_RBS_Image_HW2(I, dict_I2, map_RBS_HW2, train_init_states[i])).type(torch.float32).to(device)
        for epoch in range(nbr_epoch):
            optimizer.zero_grad()
            output = QDL_module(initial_state)
            loss = criterion(output, sample_basis_HW2)
            loss.backward()
            optimizer.step()
        for j, angle in enumerate(list(QDL_module.parameters())):
            train_encoded_parameters[i][j] = angle
    # Testing samples:
    print("Encoding testing samples:")
    for i in tqdm(range(test_init_states.shape[0])):
        sample_basis_HW2 = torch.from_numpy(map_RBS_Image_HW2(I, dict_I2, map_RBS_HW2, test_init_states[i])).type(torch.float32).to(device)
        for epoch in range(nbr_epoch):
            optimizer.zero_grad()
            output = QDL_module(initial_state)
            loss = criterion(output, sample_basis_HW2)
            loss.backward()
            optimizer.step()
        for j, angle in enumerate(list(QDL_module.parameters())):
            test_encoded_parameters[i][j] = angle 
    return(train_encoded_parameters, test_encoded_parameters)
    
    
    

