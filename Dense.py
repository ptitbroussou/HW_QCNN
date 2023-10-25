import torch 
from torch import nn
from scipy.special import binom

from toolbox import map_RBS_I2_2D, map_RBS
from RBS_Circuit import RBS_Unitaries 


#################################################################################
### Change of basis  :                                                          #
#################################################################################
def Passage_matrix_I_to_HW(I, device):
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
    Passage_matrix = torch.zeros((int(binom(2*I,2)), I**2), dtype=torch.uint8, device=device)
    mapping_input = map_RBS_I2_2D(I)
    mapping_output = map_RBS(2*I,2)
    for line in range(I):
        for column in range(I):
            Passage_matrix[mapping_output[(line, I+column)], mapping_input[(line, column+I)]] = 1
    return(Passage_matrix)


class Basis_Change_I_to_HW(nn.Module):
    """ This module allows to change the basis from the Image basis to the HW basis."""
    def __init__(self, I, device):
        """ We suppose that the input image is square and we consider no channels. """
        super().__init__()
        self.Passage_matrix = Passage_matrix_I_to_HW(I, device)

    def forward(self, input_state):
        """ This module forward a tensor made of each pure sate weighted by their
        probabilities that describe the output mixted state form the pooling layer. 
        Arg:
            - input_sate: a torch vector representing the initial input state of 
            dimension (nbr_batch, I**2).
        Output:
            - a torch vector made of several vectors that represents the output mixted
            state in the basis of HW 2. Its dimension is (nbr_batch, binom(2*I,2)).
        """
        input_state = torch.einsum('bi, oi->bo', input_state, self.Passage_matrix.to(torch.float32))
        return(input_state)


#################################################################################
### Dense Laye  :                                                               #
#################################################################################
class RBS_Dense_state_vector(nn.Module):
    """ This module describe the action of one RBS gate."""
    def __init__(self, qubit_tuple, device):
        """ Args:
            - qubit_tuple: tuple of the 2 qubits index affected by the RBS
            - device: torch device (cpu, cuda, etc...)
        """
        super().__init__()
        self.angle = nn.Parameter(torch.rand((), device=device), requires_grad=True)
        self.qubit_tuple = qubit_tuple

    def forward(self, input, RBS_unitaries):
        """ Application of the RBS corresponding unitary on the input state.
        Args:
            - input: a torch vector representing the initial input state. Its
            dimension is (nbr_batch, binom(I,2)).
            - RBS_unitaries: a dictionnary that gives the RBS unitary corresponding 
            to the qubit tuple such defined in RBS_Unitaries function. The unitary are
            of dimension (binom(I,2),binom(I,2))
        Output:
            - output state from the application of the RBS on the input state 
        """
        return(torch.einsum('bi, ii->bi', input, (RBS_unitaries[self.qubit_tuple][0]*torch.cos(self.angle) + RBS_unitaries[self.qubit_tuple][1]*torch.sin(self.angle) + RBS_unitaries[self.qubit_tuple][2])))


class Dense_RBS_state_vector(nn.Module):
    """ This module describes the action of one RBS based VQC. """
    def __init__(self, I, list_gates, device):
        """ Args:
            - I: size of the square input image
            - list_gates: list of tuples representing the qubits affected by each RBS
            - device: torch device (cpu, cuda, etc...) 
        """
        super().__init__()
        # We only store the RBS unitary corresponding to an edge in the qubit connectivity: 
        self.RBS_Unitaries_dict = RBS_Unitaries(I*2, 2, list_gates, device)
        self.RBS_gates = nn.ModuleList([RBS_Dense_state_vector(list_gates[i], device) for i in range(len(list_gates))])

    def forward(self, input_state):
        """ Feedforward of the RBS based VQC.
        Arg:
            - input_state = a state vector on which is applied the RBS from the 
            VQC. Its dimension is (nbr_batch, binom(2*I,2))
        Output:
            - final state from the application of the RBS from the VQC on the input 
            state in the basis of HW 2. Its dimension is (nbr_batch, binom(2*I,2))
        """
        for RBS in self.RBS_gates:
            input_state = RBS(input_state, self.RBS_Unitaries_dict)
        return(input_state)
