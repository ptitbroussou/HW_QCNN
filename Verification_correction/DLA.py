from numpy.linalg import matrix_rank as rank

from toolbox import *

################################################################################
### Toolbox for Dynamical Lie Algebra Calculus:
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

def Hamiltonian_bloc_FBS(edge, index_e, n, k, FBS_dictionnary, inverse_map):
    """ Given the number of qubits we are considering and the dictionnary of
    states affected by the FBS, return the bloc Hamiltonian of this FBS for the
    edge considered. """
    H_FBS = np.zeros((int(binom(n,k)),int(binom(n,k))), dtype=complex)
    a,b = edge.tuple()
    for tuple in FBS_dictionnary[index_e]:
        i,j = tuple
        H_FBS[i][j], H_FBS[j][i] = (-1)**(FBS_HW_function(a,b,j,inverse_map))*1j, (-1)**(FBS_HW_function(a,b,j,inverse_map)+1)*1j
    return(H_FBS)

def DLA_bloc_generators_RBS(k, Connectivity_Graph):
    """ Given the Connectivity Graph of the quantum circuit we are considering,
    output the Hamiltonian matrix of each one RBS on each edge, i.e. the DLA
    generators for the block matrix. """
    n = len(Connectivity_Graph.ListVertices)
    RBS_dictionnary = RBS_qubits_Graph(n, k, Connectivity_Graph)
    list_generators = []
    for edge in range(len(Connectivity_Graph.ListEdges)):
        list_generators.append(Hamiltonian_bloc_RBS(edge,n,k, RBS_dictionnary))
    return(list_generators)

def DLA_bloc_generators_FBS(k, Connectivity_Graph):
    """ Given the Connectivity Graph of the quantum circuit we are considering,
    output the Hamiltonian matrix of each one RBS on each edge, i.e. the DLA
    generators for the block matrix. """
    n = len(Connectivity_Graph.ListVertices)
    mapping = map_RBS(n,k)
    inverse_map = {v: k for k, v in mapping.items()}
    FBS_dictionnary = RBS_qubits_Graph(n, k, Connectivity_Graph) # Same rotation for RBS and FBS
    list_generators = []
    for index_e, edge in enumerate(Connectivity_Graph.ListEdges):
        list_generators.append(Hamiltonian_bloc_FBS(edge, index_e, n,k, FBS_dictionnary, inverse_map))
    return(list_generators)

################################################################################
# Toolbox to compare the evolution of the Lie algebra and the connectivity:
################################################################################
def Edges_full_connection(nbr_qubits, Qubits):
    Connections = []
    for i in range(nbr_qubits-1):
        for j in range(i+1,nbr_qubits):
            Connections.append(Edge(Qubits[i],Qubits[j]))
    return(Connections)

def Edges_line_connection(nbr_qubits, Qubits):
    Connections = []
    for i in range(nbr_qubits-1):
        Connections.append(Edge(Qubits[i],Qubits[i+1]))
    return(Connections)

################################################################################
# Builds the Lie algebra matrix from the list of control hamiltonians lHam
################################################################################
def buildLieAlgMat(H_0, lHam):
    """ We suppose that H_0 is a numpy matrix and lHam a list of numpy matrices.
    """
    n,n = lHam[0].shape
    W = np.zeros((n**2, n**2), dtype = complex)
    zeros_col = np.zeros(n**2)

    W[:,0] = H_0.flatten()
    # r tracks the number of columns
    r = 1
    # Contains the list of independant hamiltonian
    # It is equivalent to take the columns of W but more convenient to store them like that
    indepHam = [H_0]
    # We add the control hamiltonians as long as they are linearly independant
    for H in lHam:
        a = H
        W[:,r] = a.flatten()
        # If the rank is not increased by adding the column, we discard it
        if rank(W) == r:
            W[:,r] = zeros_col
        else:
            r += 1
            indepHam.append(H)
    r0 = 0
    rn = r
    # We perform any possible commutators and add them in W if it increases the rank
    while rn != r0 and rn < n**2:
        for i in range(r0, rn):
            for j in range(i):
                H = np.dot(indepHam[i], indepHam[j]) - np.dot(indepHam[j], indepHam[i])
                a = H
                W[:,r] = a.flatten()
                # If the rank is not increased by adding the column, we discard it
                if rank(W) == r:
                    W[:,r] = zeros_col
                else:
                    r += 1
                    indepHam.append(H)
                    # If the matrix is full, we return
                    if r == n**2:
                        return W
        r0 = rn
        rn = r
    return W
