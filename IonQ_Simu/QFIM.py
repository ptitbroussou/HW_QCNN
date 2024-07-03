import numpy as np
import time
from scipy.special import binom

from toolbox import CrossEntropyCost, No_Activation
from RBS_VQCs import RBS_VQC_subspace


def delta_state(VQC, input_state, output_state, theta, i, eps):
    """ This function output the partial derivative of the output state
    according to the parameter i in theta using eps Euler approximation. """
    delta_theta = theta.copy()
    delta_theta[i] += eps
    VQC.angles = delta_theta
    VQC.Update_Inner_matrices()
    delta_output = VQC.feedforward_pass(input_state).copy()
    delta_state = (delta_output - output_state)
    return(delta_state)

def QFIM_RBS_subspace(circuit_gates, input_state, n, k, theta, eps=1e-3):
    """ This funcion derive the QFIM matrix in for a RBS quantum circuit defined
    by its gates in the circuit_gates list, its number of qubits n, and its
    subspace defined by the Hamming weight k we are considering. The value eps
    is used to derive the gradient of the state.  """
    VQC = RBS_VQC_subspace(n, k, len(theta), CrossEntropyCost, No_Activation, circuit_gates)
    VQC.angles = theta.copy()
    VQC.Update_Inner_matrices()
    output_state = VQC.feedforward_pass(input_state).copy()
    # Initialization of the QFIM matrix
    QFIM = np.zeros((len(circuit_gates),len(circuit_gates)))
    list_delta_state = []
    for i in range(len(circuit_gates)):
        list_delta_state.append(delta_state(VQC, input_state, output_state, theta, i, eps))
    # We may now derive the Quantum Fisher Information Matrix
    for i in range(len(circuit_gates)):
        for j in range(len(circuit_gates)):
            delta_i_state, delta_j_state = list_delta_state[i], list_delta_state[j]
            QFIM[i][j] = 4*(np.dot(delta_i_state, delta_j_state.T) - np.dot(delta_i_state, output_state.T)*np.dot(output_state, delta_j_state.T))
    return(QFIM)
