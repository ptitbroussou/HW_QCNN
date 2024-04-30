import numpy as np
import torch
from scipy.special import binom
from itertools import combinations
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


################################################################################
# Basis change:                                                                #
################################################################################
def recursive_next_list_RBS(n, k, list_index, index):
    new_list_index = list_index.copy()
    new_list_index[index] += 1
    if (new_list_index[index] // n > 0):
        new_list_index = recursive_next_list_RBS(n - 1, k, new_list_index, index - 1)
        new_list_index[index] = new_list_index[index - 1] + 1
    return (new_list_index)


def dictionary_RBS(n, k):
    """ gives a dictionary that links the state and the list of active bits
    for a k arrangment basis """
    nbr_of_states = int(binom(n, k))
    RBS_dictionary = {}
    for state in range(nbr_of_states):
        if (state == 0):
            RBS_dictionary[state] = [i for i in range(k)]
        else:
            RBS_dictionary[state] = recursive_next_list_RBS(n, k, RBS_dictionary[state - 1], k - 1)
    return (RBS_dictionary)


def map_RBS(n, k):
    """ Given the number of qubits n and the chosen Hamming weight k, outputs
    the corresponding state for a tuple of k active qubits. """
    Dict_RBS = dictionary_RBS(n, k)
    mapping_RBS = {tuple(val): key for (key, val) in Dict_RBS.items()}
    return (mapping_RBS)


def RBS_generalized(a, b, n, k, mapping_RBS):
    """ Given the two qubits a,b the RBS gate is applied on, it outputs a list of
    tuples of basis vectors satisfying this transformation """
    # Selection of all the affected states
    RBS = []
    # List of qubits except a and b:
    list_qubit = [i for i in range(n)]
    list_qubit.pop(max(a, b))
    list_qubit.pop(min(a, b))
    # We create the list of possible active qubit set for this RBS:
    list_combinations = list(combinations(list_qubit, k - 1))
    for element in list_combinations:
        active_qubits_a = sorted([a] + list(element))
        active_qubits_b = sorted([b] + list(element))
        RBS.append((mapping_RBS[tuple(active_qubits_a)], mapping_RBS[tuple(active_qubits_b)]))
    return RBS


def map_Computational_Basis_to_HW_Subspace(n, k, map, input_state):
    """ This function transforms an input_state from the 
    computational basis to its equivalent in the basis of
    fixed Hamming Weight. 
    Args:
        - n: number of qubits
        - k: Hamming weight
        - map: a dictionary that links the active qubits and the 
        state index in the basis of fixed Hamming Weight. Can be 
        produced by using the function map_RBS.
        - input_state: a state in the computational basis.
    Output:
        - a state in the basis of fixed Hamming Weight.
    """
    output_state = torch.zeros(int(binom(n, k)))
    for key in map.keys():
        index = sum([2 ** (n - i - 1) for i in key])  # index in the computational basis:
        output_state[map[key]] = torch.real(input_state[index])  # we consider only real floats
    return (output_state)


def map_Computational_Basis_to_HW_Subspace_density(n, k, map, input_density):
    """ This function transforms an input_state from the 
    computational basis to its equivalent in the basis of
    fixed Hamming Weight. 
    Args:
        - n: number of qubits
        - k: Hamming weight
        - map: a dictionary that links the active qubits and the 
        state index in the basis of fixed Hamming Weight. Can be 
        produced by using the function map_RBS.
        - input_density: a density matrix in the computational basis.
    Output:
        - a density matrix in the basis of fixed Hamming Weight.
    """
    output_density = torch.zeros(int(binom(n, k)), int(binom(n, k)))
    for key_1 in map.keys():
        index_1 = sum([2 ** (n - i - 1) for i in key_1])  # index in the computational basis
        for key_2 in map.keys():
            index_2 = sum([2 ** (n - i - 1) for i in key_2])  # index in the computational basis
            output_density[map[key_1], map[key_2]] = torch.real(
                input_density[index_1, index_2])  # we consider only real floats
    return (output_density)


################################################################################
### RBS application in the 2D Image Basis:                                     #
################################################################################
def dictionary_RBS_I2_2D(I):
    """ gives a dictionary that links the state and the list of active bits
    for a basis of IxI images. """
    RBS_dictionary = {}
    for line in range(I):
        for column in range(I):
            RBS_dictionary[line * I + column] = [line, column + I]
    return (RBS_dictionary)


def map_RBS_I2_2D(I):
    """ Given the number of qubits n and the chosen Hamming weight k, outputs
    the corresponding state for a tuple of k active qubits. """
    Dict_RBS = dictionary_RBS_I2_2D(I)
    mapping_RBS = {tuple(val): key for (key, val) in Dict_RBS.items()}
    return (mapping_RBS)


def map_RBS_Image_HW2(I, dict_I2, map_RBS_HW2, sample):
    """ gives a dictionary that links the state in the basis of image of size 
    IxI to the equivalent state in the basis of Hamming Weight 2. """
    # dict_I2 = dictionary_RBS_I2_2D(I)
    output = np.zeros(int(binom(2 * I, 2)))
    for key in dict_I2.keys():
        (i, j) = dict_I2[key]
        index_HW2_basis = map_RBS_HW2[(i, j)]
        output[index_HW2_basis] = sample[key]
    return (output)


def RBS_generalized_I2_2D(a, b, I):
    """ Given the two qubits a,b the RBS gate is applied on, it outputs a list of
    tuples of basis vectors afffected by a rotation in the basis of IxI images. 
    We suppose that a and b are in the same register (line or column). """
    # Selection of all the affected states
    RBS = []
    if (a < I and b < I):
        # We are in the line register
        for column in range(I):
            RBS.append((a * I + column, b * I + column))
    elif (a >= I and b >= I):
        # We are in the column register
        for line in range(I):
            RBS.append((line * I + a - I, line * I + b - I))
    else:
        # We are in the cross register
        print("Error in RBS_generalized_I2: the two qubits are not in the same register")
    return (RBS)


def map_Computational_Basis_to_Image_3D_Subspace(I, J, map, input_state):
    """ This function transforms an input_state from the
    computational basis to its equivalent in the basis of
    the image. We suppose the image to be square.
    Args:
        - n: number of qubits
        - k: Hamming weight
        - map: a dictionary that links the active qubits and the
        state index in the basis of the Image. Can be produce by
        using the function map_RBS_I2_2D.
    Output:
        - a state in the basis of the image.
    """
    output_state = torch.zeros(I*I*J)
    n = I+I+J
    for key in map.keys():
        index = sum([2 ** (n - i - 1) for i in key])  # index in the computational basis:
        output_state[map[key]] = torch.real(input_state[index])  # we consider only real floats
    return (output_state)


def map_Computational_Basis_to_Image_3D_Subspace_density(I, J, map, input_density):
    """ This function transforms an input_state from the
    computational basis to its equivalent in the basis of
    the image. We suppose the image to be square.
    Args:
        - n: number of qubits
        - k: Hamming weight
        - map: a dictionary that links the active qubits and the
        state index in the basis of the Image. Can be produce by
        using the function map_RBS_I2_2D.
    Output:
        - a state in the basis of the image.
    """
    output_density = torch.zeros(I*I*J, I*I*J)
    n = I+I+J
    for key_1 in map.keys():
        index_1 = sum([2 ** (n - i - 1) for i in key_1])  # index in the computational basis
        for key_2 in map.keys():
            index_2 = sum([2 ** (n - i - 1) for i in key_2])  # index in the computational basis
            output_density[map[key_1], map[key_2]] = torch.real(
                input_density[index_1, index_2])  # we consider only real floats
    return (output_density)


def map_Computational_Basis_to_Image_Square_Subspace(n, map, input_state):
    """ This function transforms an input_state from the
    computational basis to its equivalent in the basis of
    the image. We suppose the image to be square.
    Args:
        - n: number of qubits
        - k: Hamming weight
        - map: a dictionary that links the active qubits and the
        state index in the basis of the Image. Can be produce by
        using the function map_RBS_I2_2D.
    Output:
        - a state in the basis of the image.
    """
    output_state = torch.zeros((n // 2) ** 2)
    for key in map.keys():
        index = sum([2 ** (n - i - 1) for i in key])  # index in the computational basis:
        output_state[map[key]] = torch.real(input_state[index])  # we consider only real floats
    return (output_state)


def map_Computational_Basis_to_Image_Square_Subspace_density(n, map, input_density):
    """ This function transforms an input_state from the
    computational basis to its equivalent in the basis of
    the image. We suppose the image to be square.
    Args:
        - n: number of qubits
        - k: Hamming weight
        - map: a dictionary that links the active qubits and the
        state index in the basis of the Image. Can be produce by
        using the function map_RBS_I2_2D.
    Output:
        - a state in the basis of the image.
    """
    output_density = torch.zeros((n // 2) ** 2, (n // 2) ** 2)
    for key_1 in map.keys():
        index_1 = sum([2 ** (n - i - 1) for i in key_1])  # index in the computational basis
        for key_2 in map.keys():
            index_2 = sum([2 ** (n - i - 1) for i in key_2])  # index in the computational basis
            output_density[map[key_1], map[key_2]] = torch.real(
                input_density[index_1, index_2])  # we consider only real floats
    return (output_density)


def Image_Basis_B2(I, image):
    """ This function maps a input vector image to a vector
    representing its amplitude encoding in the basis of the image. 
    Args:
        - I: size of the input image
        - image: input tensor of dimension (I,I)
    Output:
        - a tensor state in the basis of the image 
    """
    output_state = torch.zeros(I ** 2)
    for i in range(I):
        for j in range(I):
            output_state[i * I + j] = image[i][j]
    return (output_state)


################################################################################
### RBS application in the 3D Image Basis:                                     #
################################################################################
def dictionary_RBS_I2_3D(I, C):
    """ gives a dictionary that links the state and the list of active bits
    for a basis of CxIxI images. """
    RBS_dictionary = {}
    for line in range(I):
        for column in range(I):
            for channel in range(C):
                RBS_dictionary[channel * I ** 2 + line * I + column] = [line, column + I, 2 * I + channel]
    return (RBS_dictionary)


def dictionary_RBS_I2_3D_bottom(I, J):
    """ gives a dictionary that links the state and the list of active bits
    for a basis of CxIxI images. """
    RBS_dictionary = {}
    for line in range(I):
        for column in range(I):
            for channel in range(J):
                RBS_dictionary[line * I * J + column * J + channel] = [line, column + I, 2 * I + channel]
    return (RBS_dictionary)


def map_RBS_I2_3D(I, C):
    """ Given the number of qubits 2*I+C and the chosen Hamming weight 3, outputs
    the corresponding state for a tuple of k active qubits. """
    Dict_RBS = dictionary_RBS_I2_3D(I, C)
    mapping_RBS = {tuple(val): key for (key, val) in Dict_RBS.items()}
    return (mapping_RBS)


def map_RBS_I2_3D_bottom(I, C):
    """ Given the number of qubits 2*I+C and the chosen Hamming weight 3, outputs
    the corresponding state for a tuple of k active qubits. """
    Dict_RBS = dictionary_RBS_I2_3D_bottom(I, C)
    mapping_RBS = {tuple(val): key for (key, val) in Dict_RBS.items()}
    return (mapping_RBS)


def RBS_generalized_I2_3D(a, b, I, C):
    """ Given the two qubits a,b the RBS gate is applied on, it outputs a list of
    tuples of basis vectors afffected by a rotation in the basis of CxIxI images. 
    We suppose that a and b are in the same register (line or column). """
    # Selection of all the affected states
    RBS = []
    if (a < I and b < I):
        # We are in the line register
        for c in range(C):
            for column in range(I):
                RBS.append((c * I ** 2 + a * I + column, c * I ** 2 + b * I + column))
    elif (a >= I and b >= I):
        # We are in the column register
        for c in range(C):
            for line in range(I):
                RBS.append((c * I ** 2 + line * I + a - I, c * I ** 2 + line * I + b - I))
    else:
        # We are in the cross register
        print("Error in RBS_generalized_I2: the two qubits are not in the same register")
    return (RBS)


def RBS_generalized_I2_3D_bottom_channel(a, b, I, J):
    # Selection of all the affected states
    RBS = []
    if a < I and b < I:
        # We are in the line register
        for column in range(I * J):
            RBS.append((a * I * J + column, b * I * J + column))
    elif a >= 2 * I and b >= 2 * I:
        # We are in the channel register
        for rc in range(I ** 2):
            index = rc * J + (a - 2 * I)
            RBS.append((index, index + (b - a)))
    elif I <= a < 2 * I and I <= b < 2 * I:
        # We are in the column register
        for row in range(I):
            for channel in range(J):
                index = I * J * row + channel + (a - I) * J
                RBS.append((index, index + (J * (b - a))))
    else:
        # We are in the cross register
        print("Error in RBS_generalized_I2: the two qubits are not in the same register")
    return (RBS)


################################################################################
### Convolutional Quantum Neural Network:                                      #
################################################################################
def QCNN_RBS_based_VQC(I, K):
    """ I represent the size of the input image. K represents the size of the
    filter we consider. The stride is always equal to K. Each element of
    QNN_layer is a list of gates applied in parallel. Param_dictionary is a 
    dictionary that links each gate with the corresponding parameter. 
    RBS_dictionary is a dictionary that links each RBS with its corresponding
    first qubit of application. """
    # Connectivity_Graph = QCNN_Connectivity_graph(I)
    nbr_parameters = int(K * (K - 1))
    Param_dictionary, RBS_dictionary = {}, {}
    QNN_layer = [[] for i in range(2 * K - 3)]
    # QCNN circuit definition:
    for index_filter in range((I - K) // K + 1):
        # For the first half of qubits:
        PQNN_param_dictionary1, PQNN_dictionary1, PQNN_layer1 = PQNN_building_brick(K * index_filter, K,
                                                                                    index_filter * (
                                                                                            nbr_parameters // 2), 0)
        # For the second half of qubits:
        PQNN_param_dictionary2, PQNN_dictionary2, PQNN_layer2 = PQNN_building_brick(I + K * index_filter, K,
                                                                                    (I // K + index_filter) * (
                                                                                            nbr_parameters // 2),
                                                                                    (nbr_parameters // 2))
        # Updating the dictionnaries and the QNN_layers:
        RBS_dictionary.update(PQNN_dictionary1)
        RBS_dictionary.update(PQNN_dictionary2)
        Param_dictionary.update(PQNN_param_dictionary1)
        Param_dictionary.update(PQNN_param_dictionary2)
        for inner_layer_index in range(2 * K - 3):
            for element_index in range(len(PQNN_layer1[inner_layer_index])):
                QNN_layer[inner_layer_index].append(PQNN_layer1[inner_layer_index][element_index])
                QNN_layer[inner_layer_index].append(PQNN_layer2[inner_layer_index][element_index])
    return (QNN_layer, Param_dictionary, RBS_dictionary)


def QCNN_RBS_based_VQC_bottom_channel(I, K, J):
    nbr_parameters = int(K * (K - 1))
    Param_dictionary, RBS_dictionary = {}, {}
    QNN_layer = [[] for i in range(2 * K - 3)]
    # QCNN circuit definition:
    for index_filter in range(I // K):
        # For the first half of qubits:
        # PQNN_building_brick(start_qubit, size, index_first_RBS=0, index_first_param=0)
        PQNN_param_dictionary1, PQNN_dictionary1, PQNN_layer1 = PQNN_building_brick(K * index_filter, K,
                                                                                    index_filter * (
                                                                                            nbr_parameters // 2), 0)
        # For the second half of qubits:
        PQNN_param_dictionary2, PQNN_dictionary2, PQNN_layer2 = PQNN_building_brick(I + K * index_filter, K,
                                                                                    (I // K + index_filter) * (
                                                                                            nbr_parameters // 2),
                                                                                    (nbr_parameters // 2))
        # Updating the dictionnaries and the QNN_layers:
        RBS_dictionary.update(PQNN_dictionary1)
        RBS_dictionary.update(PQNN_dictionary2)
        Param_dictionary.update(PQNN_param_dictionary1)
        Param_dictionary.update(PQNN_param_dictionary2)

    for index_filter in range(1):
        # For the third half of qubits:
        PQNN_param_dictionary3, PQNN_dictionary3, PQNN_layer3 = PQNN_building_brick(2 * I + J * index_filter, J,
                                                                                    (2 * I // K + index_filter) * (
                                                                                            nbr_parameters // 2),
                                                                                    nbr_parameters)
        RBS_dictionary.update(PQNN_dictionary3)
        Param_dictionary.update(PQNN_param_dictionary3)
    return (QNN_layer, Param_dictionary, RBS_dictionary)


def PQNN_building_brick(start_qubit, size, index_first_RBS=0, index_first_param=0):
    """ This function gives back the QNN corresponding to a PQNN with nearest
    neighbours connectivity that start on qubit start_qubit. The size of the
    PQNN is given (nbr of qubits) by the input variable size.
    The PQNN_param_dictionary gives the corresponding parameters to each RBS. The
    index_first_RBS is used to named properly the RBS. The
    PQNN_dictionary is a dictionary that gives the correspondance between the
    RBS and the corresponding edge in the connectivity graph."""
    PQNN_param_dictionary, PQNN_dictionary, PQNN_layer = {}, {}, []
    List_order, List_layer_index = Pyramidal_Order_RBS_gates(size, start_qubit)
    for index, RBS in enumerate(List_order):
        PQNN_param_dictionary[index_first_RBS + index] = index_first_param + index
        PQNN_dictionary[index_first_RBS + index] = start_qubit + RBS
    # Definition of the QNN_layers thanks to List_layer_index structure
    index_RBS = index_first_RBS
    for layer in List_layer_index:
        layer_CQNN = []
        for element in layer:
            layer_CQNN.append(index_RBS)
            index_RBS += 1
        PQNN_layer.append(layer_CQNN)
    return (PQNN_param_dictionary, PQNN_dictionary, PQNN_layer)


def Pyramidal_Order_RBS_gates(nbr_qubits, first_RBS=0):
    """ This function gives the structure of each inner layer in the pyramidal
    quantum neural network. List_order gives the qubit link to each theta and
    List_layer_index gives the list of the theta for each inner layer. """
    List_layers, List_order, List_layer_index = [], [], []
    index_RBS = first_RBS
    # Beginning of the pyramid
    for i in range(nbr_qubits // 2):
        list, list_index = [], []
        for j in range(i + 1):
            if (i * 2 < (nbr_qubits - 1)):
                list.append(j * 2)
                list_index.append(index_RBS)
                index_RBS += 1
        if (len(list) > 0):
            List_layers.append(list)
            List_layer_index.append(list_index)
        list, list_index = [], []
        for j in range(i + 1):
            if (i * 2 + 1 < (nbr_qubits - 1)):
                list.append(j * 2 + 1)
                list_index.append(index_RBS)
                index_RBS += 1
        if (len(list) > 0):
            List_layers.append(list)
            List_layer_index.append(list_index)
    # End of the pyramid
    for i in range(len(List_layers) - 2, -1, -1):
        List_layers.append(List_layers[i])
        list_index = []
        for j in range(len(List_layers[i])):
            list_index.append(index_RBS)
            index_RBS += 1
        List_layer_index.append(list_index)
    # Deconcatenate:
    for i, layer in enumerate(List_layers):
        List_order += layer
    return (List_order, List_layer_index)


################################################################################
### Letao Testing Neural Network (unfinished):                                      #
################################################################################

def reduce_MNIST_dataset(data_loader, scala):
    # original data: torch.Size([60000, 28, 28])
    old_data = data_loader.dataset.data
    data_loader.dataset.data = old_data.resize_(int(data_loader.dataset.data.size(0) / scala), 28, 28)
    return data_loader


def to_density_matrix(batch_vectors, device):
    out = torch.zeros([batch_vectors.size(0), batch_vectors.size(1), batch_vectors.size(1)]).to(device)
    index = 0
    for vector in batch_vectors:
        out[index] += torch.einsum('i,j->ij', vector, vector)
        index += 1
    return out


def copy_images_bottom_channel(images, J):
    images = images.unsqueeze(1)
    upscaled_x = F.interpolate(images, size=(images.size()[-1] * J, images.size()[-1] * J), mode='nearest')
    upscaled_x = upscaled_x.squeeze(1)
    return upscaled_x


def copy_images_bottom_channel_stride(images, scale_factor):
    # Assume 'images' is a 3D torch tensor representing a batch of grayscale images
    # Dimension 0 is the batch size, 1 and 2 are both N (rows and columns respectively)
    # 'scale_factor' is the scaling factor

    # Batch and original dimensions
    batch_size, N, _ = images.shape  # Assuming square images N x N

    # New dimensions
    new_N = N * scale_factor

    # Precompute the original indices to be accessed for all images
    orig_i = torch.arange(new_N).floor_divide(scale_factor) % N
    orig_j = torch.arange(new_N).floor_divide(scale_factor) % N

    # Adjust indices to simulate the stride effect by adding a varying offset
    offset = torch.arange(new_N * new_N).view(new_N, new_N) % scale_factor
    orig_i = (orig_i.view(-1, 1) + offset) % N
    orig_j = (orig_j.view(1, -1) + offset) % N

    # Use advanced indexing to create the scaled images
    # Apply indexing directly, correctly handling the batch dimension
    scaled_images = images[:, orig_i, orig_j]

    return scaled_images


class Trace_out_dim(nn.Module):
    def __init__(self, out, device):
        super().__init__()
        self.out = out
        self.device = device

    def forward(self, input):
        # input = F.relu(input[:,-self.out:,-self.out:])
        input = input[:, -self.out:, -self.out:]
        return F.normalize(input, p=2, dim=1).to(self.device)


def map_HW_to_measure(batch_x, device):
    return torch.stack([torch.diag(x) for x in batch_x]).to(device)


def get_reduced_layers_structure(n, out):
    list_gates = []
    PQNN_param_dictionary, PQNN_dictionary, PQNN_layer = PQNN_building_brick(0, n, index_first_RBS=0,
                                                                             index_first_param=0)
    for x, y in PQNN_dictionary.items():
        list_gates.append((y, y + 1))
    list_gates.reverse()
    # print(list_gates)

    list_gates_delete = []
    PQNN_param_dictionary, PQNN_dictionary, PQNN_layer = PQNN_building_brick(0, n - out, index_first_RBS=0,
                                                                             index_first_param=0)
    for x, y in PQNN_dictionary.items():
        list_gates_delete.append((y, y + 1))
    # print(list_gates_delete)

    for e in list_gates_delete:
        list_gates.remove(e)
    list_gates.reverse()
    return list_gates


def Passage_matrix_I_to_HW_3D(I, J, k, device):
    """ This function outputs a tensor matrix that allows to pass from the
    Image basis to the HW basis. We assume to consider square images with no
    channels.
    Args:
        - I: size of the input image
        - device: torch device (cpu, cuda, etc...)
    Output:
        - Passage_matrix: tensor matrix of size (int(binom(2*I,2)), I**2) that allows
        to pass from the Image basis to the HW basis.
    """
    Passage_matrix = torch.zeros((int(binom(I + I + J, k)), I * I * J), dtype=torch.uint8, device=device)
    mapping_input = map_RBS_I2_3D_bottom(I, J)
    mapping_output = map_RBS(I + I + J, k)
    for line in range(I):
        for column in range(I):
            for channel in range(J):
                # print("line: " + str(line) + ", " + str(I+column) + ", " + str(2*I+channel) )
                output_index = mapping_output[(line, I + column, 2 * I + channel)]
                intput_index = mapping_input[(line, I + column, 2 * I + channel)]
                Passage_matrix[output_index, intput_index] = 1
    return (Passage_matrix)


class Basis_Change_I_to_HW_density_3D(nn.Module):
    """ This module allows to change the basis from the Image basis to the HW basis."""

    def __init__(self, I, J, k, device):
        """ We suppose that the input image is square and we consider no channels. """
        super().__init__()
        self.Passage_matrix = Passage_matrix_I_to_HW_3D(I, J, k, device).to(torch.float)

    def forward(self, input_state):
        """ This module forward a tensor made of each pure sate weighted by their
        probabilities that describe the output mixted state form the pooling layer.
        Arg:
            - input: a torch vector representing the initial input state. Its
            dimension is (nbr_batch, I**2, I**2).
        Output:
            - a torch density operator that represents the output mixted state in
            the basis of HW 2. Its dimension is (nbr_batch, binom(2*I,2), binom(2*I,2)).
        """
        return self.Passage_matrix @ input_state @ self.Passage_matrix.T


def get_predict_number_vector(output_network, device):
    """
    input: 5*91*91， 10*1540*1540
    output:5*10
    """
    batch_number = output_network.size()[0]
    matrix_size = output_network.size()[1]
    step = output_network.size()[1] // 10
    batch_output = []
    for i in range(batch_number):
        diagonal = torch.diag(output_network[i]).to(device)
        output_list = []
        for j in range(0, matrix_size, step):
            chunk_sum = torch.sum(diagonal[j:j + step]).to(device)
            output_list.append(chunk_sum)
        batch_output.append(torch.tensor(output_list).to(device))
    MO = torch.stack(batch_output).to(device)
    return MO


def softargmax(x, device):
    beta = 1.0
    xx = beta * x
    sm = torch.nn.functional.softmax(xx, dim=-1)
    indices = torch.arange(len(x)).to(device)
    y = torch.mul(indices, sm)
    result = torch.sum(y).to(device)
    return result


def batch_softargmax(predict_number_vectors, device):
    """
    input: batch * 10 tensor, e.g. 5*10
    output: batch * 1 tensor
    """
    out = torch.zeros(predict_number_vectors.size()[0], requires_grad=True).to(device)
    index = 0
    for vector in predict_number_vectors:
        out[index] += softargmax(vector, device)
        index += 1
    return out.float()


def get_batch_projectors(numbers, batch_size, CN2, device):
    """
    get the target matrix to calculate the loss function
    numbers: target in the MNIST dataloader, size: batch * 1
    CN2: binom(n,2)
    output size: batch * CN2 * CN2
    """
    output = torch.zeros(batch_size, CN2, CN2).to(device)
    projector_size = CN2 // 10
    for i in range(batch_size):
        for j in range(numbers[i] * projector_size, (numbers[i] + 1) * projector_size):
            output[i][j][j] += 1.0 / projector_size
    return output


def get_batch_dot_projectors(numbers, batch_size, CN2, device):
    """
    get the dot target matrix to calculate the loss function
    numbers: target in the MNIST dataloader, size: batch * 1
    CN2: binom(n,2)
    output size: batch * CN2 * CN2
    """
    output = torch.zeros(batch_size, CN2, CN2).to(device)
    projector_size = CN2 // 10
    for i in range(batch_size):
        output[i][((2 * numbers[i] + 1) // 2) * projector_size][((2 * numbers[i] + 1) // 2) * projector_size] += 1.0
    return output


def filter_dataloader(dataloader, classes=[0, 1]):
    filtered_data = []
    filtered_targets = []

    for data, target in dataloader:
        mask = target == classes[0]
        for c in classes[1:]:
            mask = mask | (target == c)

        filtered_data.append(data[mask])
        filtered_targets.append(target[mask])

    # Concatenate all collected data and targets
    filtered_data = torch.cat(filtered_data, dim=0)
    filtered_targets = torch.cat(filtered_targets, dim=0)

    # Create a new TensorDataset and DataLoader from the filtered data and targets
    filtered_dataset = TensorDataset(filtered_data, filtered_targets)
    filtered_dataloader = DataLoader(filtered_dataset, batch_size=dataloader.batch_size, shuffle=True)

    return filtered_dataloader


def get_full_pyramid_gates(n):
    list_gates = []
    _, PQNN_dictionary, _ = PQNN_building_brick(0, 5, index_first_RBS=0, index_first_param=0)
    for x, y in PQNN_dictionary.items():
        list_gates.append((y,y+1))
    return list_gates