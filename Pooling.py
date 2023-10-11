import torch 
from torch import nn
from scipy.special import binom

from toolbox import map_RBS_I2_2D, map_RBS_I2_3D, map_RBS



def Pooling_2D_Projector(I, O, device):
    """ This function mimics the effect of a HW preserving pooling layer on half
    the remaining qubits. We suppose that the input image is square.
    Args:
        - I: size of the input image
        - O: size of the output image (we suppose O = I//2 for now)
        - device: torch device (cpu, cuda, etc...)
    """
    # We load a map that link each index in the input basis with qubits in state |1>
    map_I = map_RBS_I2_2D(I)
    # We load a map that link each index in the output basis with qubits in state |1>
    map_O = map_RBS_I2_2D(O)
    # We consider the case where all measured qubits are on state |0>:
    Projector = torch.zeros((O**2, I**2), dtype=torch.uint8, device=device)
    for key in map_I.keys():
        i,j = key
        if ((i%2 == 0) and (j%2 == 0)):
            Projector[map_O[(i//2, j//2)], map_I[key]] = 1
    # We consider the case where we measured 2 qubits in state |1>:
    Projectors, index = torch.zeros(((I//2)**2, O**2, I**2), dtype=torch.uint8, device=device), 0
    for i in range(O):
        for j in range(O):
            Projectors[index, i*O+2*j+1, i*2*I+j*2] = 1
            index += 1
    # We finally consider the case where we measured only one qubit in state |1>:
    Projectors2, index = torch.zeros((I, O**2, I**2), dtype=torch.uint8, device=device), 0
    for i in range(O):
        for j in range(O):
            Projectors2[index, i*O+j]
        index +=1
        








################################################################################
### 2 Dimensional Pooling Layers:                                              #
################################################################################
### Pooling between Convolutional Layers:                                        #
def Pooling_Matrix_2D(I, O, device):
    """ This function return the matrix that allows to pool the images in the basis
    of IxI images (unary basis on line and column registers). """
    Transformation_matrix = torch.zeros(O**2, I**2, dtype=torch.uint8, device=device)
    Pooling_window_size = I//O
    mapping_input = map_RBS_I2_2D(I)
    mapping_output = map_RBS_I2_2D(O)
    for line in range(O):
        for column in range(O):
            for i in range(line*Pooling_window_size,(line+1)*Pooling_window_size):
                for j in range(column*Pooling_window_size,(column+1)*Pooling_window_size):
                    Transformation_matrix[mapping_output[(line, O+column)], mapping_input[(i, j+I)]] = 1
    return(Transformation_matrix)

def sqrt_with_identity(x):
    # Check if x is zero, if yes, return x itself, otherwise return sqrt(x)
    return torch.sqrt(torch.where(x > 0, x, torch.zeros_like(x)))


class Custom_Pooling_2D(nn.Module):
    """ This module describe an Average Pooling layer effect on a convolutional
    architecture made of HW preserving quantum circuits. """
    def __init__(self, I, O, device):
        """ The input Images are of size I*I and the pooling reduce the images
        into size O*O. """
        super().__init__()
        self.Transformation_matrix = Pooling_Matrix_2D(I, O, device)

    def forward(self, input_state):
        input_state = input_state.unsqueeze(-1)
        input_state = input_state**2
        # Not sure that tensor casting is the more optimal way to do this matrix multiplication
        input_state = torch.matmul(self.Transformation_matrix.to(torch.float32), input_state)
        #input_state = sqrt_with_identity(input_state)
        input_state = input_state.squeeze(-1)
        return(input_state)
    
### Final Pooling between a Convolutional Layer and a Dense:                     #
def Pooling_Matrix_Change_Basis_2D(I, O, device):
    """ This function return the matrix that allows to pool the images in the basis
    of IxI images (unary basis on line and column registers). """
    Transformation_matrix = torch.zeros(int(binom(2*O,2)), I**2, dtype=torch.uint8, device=device)
    Pooling_window_size = I//O
    mapping_input = map_RBS_I2_2D(I)
    mapping_output = map_RBS(2*O,2)
    for line in range(O):
        for column in range(O):
            for i in range(line*Pooling_window_size,(line+1)*Pooling_window_size):
                for j in range(column*Pooling_window_size,(column+1)*Pooling_window_size):
                    Transformation_matrix[mapping_output[(line, O+column)], mapping_input[(i, j+I)]] = 1
    return(Transformation_matrix)

class Custom_Final_Pooling_2D(nn.Module):
    """ This module describe an Average Pooling layer effect on a convolutional
    architecture made of HW preserving quantum circuits. """
    def __init__(self, I, O, device):
        """ The input Images are of size I*I and the pooling reduce the images
        into size O*O. """
        super().__init__()
        self.Transformation_matrix = Pooling_Matrix_Change_Basis_2D(I, O, device)

    def forward(self, input_state):
        input_state = input_state.unsqueeze(-1)
        input_state = input_state**2
        # Not sure that tensor casting is the more optimal way to do this matrix multiplication
        input_state = torch.matmul(self.Transformation_matrix.to(torch.float32), input_state)
        #input_state = sqrt_with_identity(input_state)
        input_state = input_state.squeeze(-1)
        return(input_state)
    



################################################################################
### 3 Dimensional Pooling Layers:                                              #
################################################################################
### Pooling between Convolutional Layers:                                        #
def Pooling_Matrix_3D(I, C, O, device):
    """ This function return the matrix that allows to pool the images in the basis
    of CxIxI images (unary basis on line and column registers). """
    Transformation_matrix = torch.zeros(C*O**2, C*I**2, dtype=torch.uint8, device=device)
    Pooling_window_size = I//O
    mapping_input = map_RBS_I2_3D(I,C)
    mapping_output = map_RBS_I2_3D(O,C)
    for line in range(O):
        for column in range(O):
            for c in range(C):
                for i in range(line*Pooling_window_size,(line+1)*Pooling_window_size):
                    for j in range(column*Pooling_window_size,(column+1)*Pooling_window_size):
                        Transformation_matrix[mapping_output[(line, O+column, 2*O+c)], mapping_input[(i, j+I, 2*I+c)]] = 1
    return(Transformation_matrix)


class Custom_Pooling_3D(nn.Module):
    """ This module describe an Average Pooling layer effect on a convolutional
    architecture made of HW preserving quantum circuits. """
    def __init__(self, I, C, O, device):
        """ The input Images are of size I*I and the pooling reduce the images
        into size O*O. """
        super().__init__()
        self.Transformation_matrix = Pooling_Matrix_3D(I, C, O, device)

    def forward(self, input_state):
        input_state = input_state.unsqueeze(-1)
        input_state = input_state**2
        # Not sure that tensor casting is the more optimal way to do this matrix multiplication
        input_state = torch.matmul(self.Transformation_matrix.to(torch.float32), input_state)
        input_state = sqrt_with_identity(input_state)
        input_state = input_state.squeeze(-1)
        return(input_state)
    
### Final Pooling between a Convolutional Layer and a Dense:                     #
def Pooling_Matrix_Change_Basis_3D(I, C, O, device):
    """ This function return the matrix that allows to pool the images in the basis
    of IxI images (unary basis on line and column registers). """
    Transformation_matrix = torch.zeros(int(binom(2*O+C,3)), C*I**2, dtype=torch.uint8, device=device)
    Pooling_window_size = I//O
    mapping_input = map_RBS_I2_3D(I,C)
    mapping_output = map_RBS(2*O+C,3)
    for line in range(O):
        for column in range(O):
            for c in range(C):
                for i in range(line*Pooling_window_size,(line+1)*Pooling_window_size):
                    for j in range(column*Pooling_window_size,(column+1)*Pooling_window_size):
                        Transformation_matrix[mapping_output[(line, O+column, 2*O+c)], mapping_input[(i, j+I, 2*I+c)]] = 1
    return(Transformation_matrix)

class Custom_Final_Pooling_3D(nn.Module):
    """ This module describe an Average Pooling layer effect on a convolutional
    architecture made of HW preserving quantum circuits. """
    def __init__(self, I, C, O, device):
        """ The input Images are of size I*I and the pooling reduce the images
        into size O*O. """
        super().__init__()
        self.Transformation_matrix = Pooling_Matrix_Change_Basis_3D(I, C, O, device)

    def forward(self, input_state):
        input_state = input_state.unsqueeze(-1)
        input_state = input_state**2
        # Not sure that tensor casting is the more optimal way to do this matrix multiplication
        input_state = torch.matmul(self.Transformation_matrix.to(torch.float32), input_state)
        input_state = sqrt_with_identity(input_state)
        input_state = input_state.squeeze(-1)
        return(input_state)
    

