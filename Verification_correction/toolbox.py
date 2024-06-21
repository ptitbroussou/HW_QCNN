import numpy as np
from scipy.special import binom
from itertools import combinations

################################################################################
# Hamming weight preserving quantum circuit:
################################################################################
def recursive_next_list_RBS(n, k, list_index, index):
    #print(n, k, list_index, index)
    new_list_index = list_index.copy()
    new_list_index[index] += 1
    if (new_list_index[index]//n > 0):
        new_list_index = recursive_next_list_RBS(n-1,k, new_list_index, index - 1)
        new_list_index[index] = new_list_index[index - 1] + 1
    return(new_list_index)

def dictionnary_RBS(n,k):
    """ gives a dictionnary that links the state and the list of active bits
    for a k arrangment basis """
    nbr_of_states = int(binom(n,k))
    RBS_dictionnary = {}
    for state in range(nbr_of_states):
        if (state == 0):
            RBS_dictionnary[state] = [i for i in range(k)]
        else:
            RBS_dictionnary[state] = recursive_next_list_RBS(n, k, RBS_dictionnary[state-1], k-1)
    return(RBS_dictionnary)

def map_RBS(n, k):
    """ Given the number of qubits n and the chosen Hamming weight k, outputs
    the corresponding state for a tuple of k active qubits. """
    Dict_RBS = dictionnary_RBS(n,k)
    mapping_RBS = {tuple(val): key for (key,val) in Dict_RBS.items()}
    return(mapping_RBS)

def RBS_generalized(a, b, n, k, mapping_RBS):
    """ given the two qubits a,b the RBS gate is applied on, it outputs a list of
    tuples of basis vectors satisfying this transformation """
    # Selection of all the affected states
    RBS = []
    # List of qubits except a and b:
    list_qubit = [i for i in range(n)]
    list_qubit.pop(max(a,b))
    list_qubit.pop(min(a,b))
    # We create the list of possible active qubit set for this RBS:
    list_combinations = list(combinations(list_qubit, k-1))
    for element in list_combinations:
        active_qubits_a = sorted([a] + list(element))
        active_qubits_b = sorted([b] + list(element))
        RBS.append((mapping_RBS[tuple(active_qubits_a)], mapping_RBS[tuple(active_qubits_b)]))
    return RBS

def FBS_HW_function(i,j,state, inverse_map):
    """ given a |state> and two qubit i and j, derive the Hamming weight of the
    (j-i)-binary word corresponding to the state[i:j]. """
    result = 0
    for qubit in inverse_map[state]:
        if ((qubit > min(i,j)) and (qubit < max(i,j))):
            result += 1
    return(result)


def RBS_qubits_Graph(nbr_qubits, k, Connectivity_Graph):
    """ Given a graph of the connectivityn output a dictionnary that link each
    edge with the list of rotation corresponding to the application of one RBS
    on this edge. """
    mapping_RBS = map_RBS(nbr_qubits, k)
    RBS_qubits = {i:RBS_generalized(Connectivity_Graph.ListEdges[i].tuple()[0], Connectivity_Graph.ListEdges[i].tuple()[1], nbr_qubits, k, mapping_RBS) for i in range(len(Connectivity_Graph.ListEdges))}
    return(RBS_qubits)

################################################################################
#### Cost Functions and Activation Functions:
################################################################################
def ReLu_prime(z):
    """Derivative of the ReLu function."""
    z_copy = z.copy()
    for i in range(z_copy.shape[0]):
        if z_copy[i]<= 0.0:
            z_copy[i]=0.0
        else:
            z_copy[i]=1.0
    return z_copy

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2
    @staticmethod
    def delta(z_prime, a, y):
        """Return the error delta from the output layer."""
        return 2*(a-y)* z_prime

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z_prime, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z_prime`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)

class SigmoidActivation(object):
    @staticmethod
    def fn(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))
    def prime(z):
        """Derivative of the sigmoid function."""
        return np.exp(-z)/(1.0+np.exp(-z))**2

class ReLuActivation(object):
    @staticmethod
    def fn(z):
        """The ReLu function."""
        return np.maximum(0.0,z)
    def prime(z):
        """Derivative of the ReLu function."""
        for i in range(z.shape[0]):
            if z[i]<= 0.0:
                z[i]=0.0
            else:
                z[i]=1.0
        return z

class No_Activation(object):
    @staticmethod
    def fn(z):
        return(z)
    def delta(z):
        return(1)
    def prime(z):
        for i in range(z.shape[0]):
            z[i] = 1
        return z


################################################################################
### Connectivity as a Graph:
################################################################################
class Network():
    """ network defined by its vertices and edges """
    def __init__(self,ListVertices,ListEdges):
        self.ListVertices = ListVertices
        self.ListEdges = ListEdges
    def __str__(self):
        return("Vertices of the Network:{} \nEdges of the Network {}".format(self.ListVertices,self.ListEdges))
    def AddVertex(self,vertex):
        self.ListVertices = self.ListVertices + [vertex]
    def AddEdges(self,edge):
        self.ListEdges = self.ListEdges + [edge]
    def __repr__(self):
        return(str(self))

class Vertex():
    """ vertex defined by name and edges """
    def __init__(self,Name, Id): # Name as str, Id as int
        self.Name = Name
        self.Id = Id
    def __str__(self):
        return("{}".format(self.Name))
    def __repr__(self):
        return(str(self))

class Edge():
    """ edge defined by start and end vertices """
    def __init__(self,StartVertex,EndVertex): # vertex and vertex
        self.StartVertex = StartVertex
        self.EndVertex = EndVertex
    def tuple(self):
        return((self.StartVertex.Id, self.EndVertex.Id))
    def __str__(self):
        return("{}<-->{}".format(self.StartVertex,self.EndVertex))
    def __repr__(self):
        return(str(self))

################################################################################
### Dynamical Lie Algebra Calculus:
################################################################################
def complete_bitstring(bitstring, n):
    """ Given a bitstring and the number of qubit considered n, outputs a padded
    bitstring such as the lenght of the output is equal to n+2 """
    if (len(bitstring) < (n+2)):
        pad = ''
        for i in range(n+2 - len(bitstring)):
            pad += '0'
        new_bitstring = bitstring[:2] + pad + bitstring[2:]
    else:
        new_bitstring = bitstring
    return(new_bitstring)

def RBS_states_affected(a,b,n):
    """ Given the index of two qubits connected by a RBS in a n-qubit quantum
    circuit, outputs the list of states affected as tuples. We must have a > b
    and n > 2. """
    list_tuple_res = []
    list_possible_bitstring = [complete_bitstring(bin(i),n-2) for i in range(2**(n-2))]
    for bitstring in list_possible_bitstring:
        state_i, state_j = bitstring, bitstring
        state_i = state_i[:n+1-b] + '0' + state_i[n+1-b:]
        state_i = state_i[:n+1-a] + '1' + state_i[n+1-a:]
        state_j = state_j[:n+1-b] + '1' + state_j[n+1-b:]
        state_j = state_j[:n+1-a] + '0' + state_j[n+1-a:]
        list_tuple_res.append((int(state_i,2), int(state_j,2)))
    return(list_tuple_res)

def Hamiltonian_RBS(a,b,n):
    """ Given the number of qubits we are considering and the two qubits
    affected by the RBS, return the Hamiltonian of this RBS in the Hilbert space
    of our circuit. """
    H_RBS = np.zeros((2**n,2**n), dtype=complex)
    list_states_RBS = RBS_states_affected(a,b,n)
    for tuple in list_states_RBS:
        i,j = tuple
        H_RBS[i][j], H_RBS[j][i] = 1j, -1j
    return(H_RBS)

def DLA_generators(Connectivity_Graph):
    """ Given the Connectivity Graph of the quantum circuit we are considering,
    output the Hamiltonian matrix of each one RBS on each edge, i.e. the DLA
    generators. """
    n = len(Connectivity_Graph.ListVertices)
    list_generators = []
    for edge in Connectivity_Graph.ListEdges:
        a,b = edge.tuple()
        list_generators.append(Hamiltonian_RBS(a,b,n))
    return(list_generators)

def Hamiltonian_bloc_RBS(edge, n, k, RBS_dictionnary):
    """ Given the number of qubits we are considering and the dictionnary of
    states affected by the RBS, return the bloc Hamiltonian of this RBS for the
    edge considered. """
    H_RBS = np.zeros((int(binom(n,k)),int(binom(n,k))), dtype=complex)
    for tuple in RBS_dictionnary[edge]:
        i,j = tuple
        H_RBS[i][j], H_RBS[j][i] = 1j, -1j
    return(H_RBS)

def DLA_bloc_generators(k, Connectivity_Graph):
    """ Given the Connectivity Graph of the quantum circuit we are considering,
    output the Hamiltonian matrix of each one RBS on each edge, i.e. the DLA
    generators for the block matrix. """
    n = len(Connectivity_Graph.ListVertices)
    RBS_dictionnary = RBS_qubits(n, k, Connectivity_Graph)
    list_generators = []
    for edge in range(len(Connectivity_Graph.ListEdges)):
        list_generators.append(Hamiltonian_bloc_RBS(edge,n,k, RBS_dictionnary))
    return(list_generators)
