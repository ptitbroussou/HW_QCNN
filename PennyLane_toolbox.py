import pennylane as qml

from toolbox import QCNN_RBS_based_VQC, Pyramidal_Order_RBS_gates


# RBS simulation via PennyLane toolbox:
def RBS(i,j,angle):
    """ This function apply a RBS on the PennyLane quantum circuit. The RBS is 
    decomposed to match the PennyLane library.
    Args:
        - i,j: qubits on which we apply the RBS
        - angle: parameter of the RBS
    """
    qml.Hadamard(wires=i)
    qml.Hadamard(wires=j)
    qml.CZ(wires=[i,j])
    qml.RY(-angle, wires=i)
    qml.RY(angle, wires=j)
    qml.CZ(wires=[i,j])
    qml.Hadamard(wires=i)
    qml.Hadamard(wires=j)

# Convolutional layer definition:
def Conv_2D_gates(affected_qubits, K):
    """ This function creates a Convolutional layer with RBS gates. We suppose
    that the image is square.
    Args:
        - affected_qubits: list that contains the qubits on which we apply 
    the RBS gates
        - K: size of the filter window
        
    The list affected_qubits contains the qubits on which we apply 
    the RBS gates, and K represents the size of the filter window.
     angles is a list of parameters
    of size K*(K-1). """
    list_gates = []
    _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC(len(affected_qubits)//2, K)
    for key in RBS_dictionary:
            list_gates.append((affected_qubits[RBS_dictionary[key]], affected_qubits[RBS_dictionary[key]+1]))
    return(Param_dictionary, list_gates)

# Convolutional layer using pennylane:
def Conv_RBS_2D(angles, Param_dictionary, list_gates):
    """ This function creates a Convolutional layer with RBS gates.
    The list affected_qubits contains the qubits on which we apply 
    the RBS gates, and K represents the size of the filter window.
    We suppose that the image is square. angles is a list of parameters
    of size K*(K-1). """
    for index, RBS in enumerate(list_gates):
        i,j = RBS
        angle = angles[Param_dictionary[index]]
        # decomposing the RBS gate:
        qml.Hadamard(wires=i)
        qml.Hadamard(wires=j)
        qml.CZ(wires=[i,j])
        qml.RY(-angle, wires=i)
        qml.RY(angle, wires=j)
        qml.CZ(wires=[i,j])
        qml.Hadamard(wires=i)
        qml.Hadamard(wires=j)   

#############################################################################################################
# Pooling layers:
#############################################################################################################
# Pooling layer definition:
def Pool_2D(affected_qubits):
    """ This function apply a quantum circuit in the remaining qubits
    of the circuit (in affected_qubits) made of CNOTS. We suppose that
    the number of qubits in affected_qubits is even."""
    for i in range(len(affected_qubits)//2):
        qml.CNOT(wires=[affected_qubits[2*i],affected_qubits[2*i+1]])

#############################################################################################################
# One layer (Convolution + Pooling):
#############################################################################################################
def conv_and_pooling(angles, K, affected_qubits):
    """ Apply both the convolution and the pooling layer."""
    conv2D_dictionary, conv2D_gates = Conv_2D_gates(affected_qubits, K)
    Conv_RBS_2D(angles, conv2D_dictionary, conv2D_gates)
    Pool_2D(affected_qubits)

#############################################################################################################
# Fully connected layers:
#############################################################################################################
# Fully connected layer definition:
def denseRBS_gates(affected_qubits):
    """ This function apply a quantum circuit in the remaining qubits
    of the circuit (in affected_qubits) made of RBS."""
    List_PQNN, _ = Pyramidal_Order_RBS_gates(len(affected_qubits))
    list_gates = []
    for index in List_PQNN:
        list_gates.append((affected_qubits[index], affected_qubits[index+1]))
    return(list_gates)

def dense_RBS(angles, list_gates):
    """ This function apply a RBS dense layer in the remaining qubits """
    #list_gates = denseRBS_gates(affected_qubits)
    for index, RBS in enumerate(list_gates):
        i,j = RBS
        angle = angles[index]
        # decomposing the RBS gate:
        qml.Hadamard(wires=i)
        qml.Hadamard(wires=j)
        qml.CZ(wires=[i,j])
        qml.RY(-angle, wires=i)
        qml.RY(angle, wires=j)
        qml.CZ(wires=[i,j])
        qml.Hadamard(wires=i)
        qml.Hadamard(wires=j)
    
