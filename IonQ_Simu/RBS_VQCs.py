import numpy as np
from tqdm import tqdm
from scipy.special import binom
import matplotlib.pyplot as plt

from toolbox import CrossEntropyCost, ReLuActivation, QuadraticCost, No_Activation
from toolbox import map_RBS, RBS_generalized


# Useful functions to define our RBS based Hamming weight preserving circuit:
def Random_Particle_Preserving_circuit(nbr_qubit, k, nbr_parameters):
    """ Define a random particle preserving quantum circuit  """
    QNN_layer, QNN_dictionnary = [[i] for i in range(nbr_parameters)], {}
    mapping_RBS = map_RBS(nbr_qubit, k)
    for RBS_index in range(nbr_parameters):
        i,j = np.random.choice([i for i in range(nbr_qubit)], 2, False)
        QNN_dictionnary[RBS_index] = RBS_generalized(i, j, nbr_qubit, k, mapping_RBS)
    return(QNN_dictionnary, QNN_layer)

def Chosen_Particle_Preserving_circuit(n,k, list_gates):
    """ Design the particle preserving quantum circuit according to the list_gates,
    a list of tuples that describe the qubits on which are applied the RBS. """
    QNN_layer, QNN_dictionnary = [[i] for i in range(len(list_gates))], {}
    mapping_RBS = map_RBS(n, k)
    for RBS in range(len(QNN_layer)):
        i,j = list_gates[RBS]
        QNN_dictionnary[RBS] = RBS_generalized(i,j, n, k, mapping_RBS)
    return(QNN_dictionnary, QNN_layer)



class RBS_VQC_subspace(object):
    def __init__(self, n, k, nbr_parameters, cost=QuadraticCost, activation=No_Activation, list_gates = []):
        """ The input_size gives you the number of qubits. The output_size gives
        the number of qubits considered at the end of the QNN.  """
        self.nbr_qubit = n
        self.hamming_weight = k
        self.nbr_states = int(binom(n,k))
        self.cost = cost
        self.activation = activation
        self.angles = 2*np.pi*np.random.rand(nbr_parameters)
        if (len(list_gates) != nbr_parameters):
            self.QNN_dictionnary, self.QNN_layers = Random_Particle_Preserving_circuit(n,k, nbr_parameters)
        else:
            self.QNN_dictionnary, self.QNN_layers = Chosen_Particle_Preserving_circuit(n,k, list_gates)
        self.Inner_layers, self.Inner_errors = np.zeros((len(self.QNN_layers) + 1, self.nbr_states)), np.zeros((len(self.QNN_layers) + 1, self.nbr_states))
        self.Inner_layer_matrices, self.Inner_error_matrices = np.array([np.eye(self.nbr_states) for i in range(len(self.QNN_layers))]), np.array([np.eye(self.nbr_states) for i in range(len(self.QNN_layers))])

    # Updating the Inner_layer_matrices and Inner_error_matrices:
    def Update_Inner_matrices(self):
        """ Udpate the inner layer matrices and the inner error matrices
        according to the value of the angles. This need to be done after each
        udpate of the angles. """
        for index_inner_layer, inner_layer in enumerate(self.QNN_layers):
            for RBS_gate in inner_layer:
                theta = self.angles[RBS_gate]
                for tuple_states in self.QNN_dictionnary[RBS_gate]:
                    i,j = tuple_states
                    # Inner layer matrices:
                    self.Inner_layer_matrices[index_inner_layer][i][i] = np.cos(theta)
                    self.Inner_layer_matrices[index_inner_layer][j][j] = np.cos(theta)
                    self.Inner_layer_matrices[index_inner_layer][i][j] = np.sin(theta)
                    self.Inner_layer_matrices[index_inner_layer][j][i] = -np.sin(theta)
                    # Inner error matrices:
                    self.Inner_error_matrices[index_inner_layer][i][i] = np.cos(theta)
                    self.Inner_error_matrices[index_inner_layer][j][j] = np.cos(theta)
                    self.Inner_error_matrices[index_inner_layer][i][j] = -np.sin(theta)
                    self.Inner_error_matrices[index_inner_layer][j][i] = np.sin(theta)

    # Feedforward and backpropagation:
    def feedforward_pass(self, input_vector):
        """ Update the inner layers of the network for this input vector. """
        self.Inner_layers[0] = input_vector
        for index_inner_layer in range(len(self.QNN_layers)):
            self.Inner_layers[index_inner_layer + 1] = np.dot(self.Inner_layer_matrices[index_inner_layer], self.Inner_layers[index_inner_layer])
        return(self.activation.fn(self.Inner_layers[-1]))

    def backpropagation(self, input_error):
        """ Update the inner errors of the network for this input error. """
        self.Inner_errors[-1] = input_error
        for index_inner_layer in range(len(self.QNN_layers)):
            backprop_index = len(self.QNN_layers) - index_inner_layer - 1
            self.Inner_errors[backprop_index] = np.dot(self.Inner_error_matrices[backprop_index], self.Inner_errors[backprop_index+1])
        return(self.Inner_errors[0])

    # Training functions:
    def gradient_derivation(self):
        """ Gives the gradient of the QNN variationnal parameters. """
        gradient = np.zeros(np.shape(self.angles))
        for index_inner_layer, inner_layer in enumerate(self.QNN_layers):
            for RBS_gate in inner_layer:
                theta = self.angles[RBS_gate]
                for tuple_states in self.QNN_dictionnary[RBS_gate]:
                    i,j = tuple_states
                    gradient[RBS_gate] += self.Inner_errors[index_inner_layer+1][i]*(-np.sin(theta)*self.Inner_layers[index_inner_layer][i] + np.cos(theta)*self.Inner_layers[index_inner_layer][j])
                    gradient[RBS_gate] += self.Inner_errors[index_inner_layer+1][j]*(-np.cos(theta)*self.Inner_layers[index_inner_layer][i] - np.sin(theta)*self.Inner_layers[index_inner_layer][j])
        return(gradient)

    def cost_accuracy_derivation(self, X, Y, nbr_class):
        results = [(np.argmax(self.feedforward_pass(X[i])[:nbr_class]), np.argmax(Y[i])) for i in range(np.shape(X)[0])]
        result_accuracy = sum(int(x == y) for (x, y) in results)/(np.shape(X)[0])
        result_cost = sum([self.cost.fn(self.feedforward_pass(X[i])[0], Y[i]) for i in range(np.shape(X)[0])])/(np.shape(X)[0])
        return (result_cost, 100*result_accuracy)

    def SGD_backprop(self, X_train, Y_train, X_test, Y_test, plot_training = False, nbr_class=2, batch_size = 5, max_epochs = 100, eps=0, learning_rate = 1, regularization = 0):
        """ Apply the backpropagation formula and update the list of codes of
        each optimization iteration with SGD algorithm. """
        self.Update_Inner_matrices()
        nbr_batches = np.shape(X_train)[0]//batch_size
        List_index_samples = [i for i in range(np.shape(X_train)[0])]
        if plot_training:
            Training_acc, Testing_acc = [], []
            _, training_acc = self.cost_accuracy_derivation(X_train, Y_train, nbr_class)
            _, testing_acc = self.cost_accuracy_derivation(X_test, Y_test, nbr_class)
            Training_acc.append(training_acc)
            Testing_acc.append(testing_acc)
        for epoch in tqdm(range(max_epochs)):
            np.random.shuffle(List_index_samples) # We shuffle the samples before each training epoch
            for i in range(nbr_batches): # Gradient derivation
                gradients = np.zeros(np.shape(self.angles)[0])
                for j in range(batch_size):
                    self.feedforward_pass(X_train[List_index_samples[i*batch_size + j]])
                    input_error = self.cost.delta(self.activation.prime(self.Inner_layers[-1]), self.activation.fn(self.Inner_layers[-1]), Y_train[List_index_samples[i*batch_size + j]])
                    self.backpropagation(input_error)
                    nabla_iteration = self.gradient_derivation()
                    gradients = gradients + nabla_iteration/batch_size #nbr_batches
                self.angles = self.angles - regularization*learning_rate*self.angles -  learning_rate*gradients
                self.Update_Inner_matrices()
            if plot_training:
                _, training_acc = self.cost_accuracy_derivation(X_train, Y_train, nbr_class)
                _, testing_acc = self.cost_accuracy_derivation(X_test, Y_test, nbr_class)
                Training_acc.append(training_acc)
                Testing_acc.append(testing_acc)
        if plot_training:
            fig = plt.figure()
            plt.plot([i for i in range(len(Training_acc))], Training_acc, 'b', label = 'Training accuracy')
            plt.plot([i for i in range(len(Testing_acc))], Testing_acc, 'r', label = 'Testing accuracy')
            plt.title('')
            plt.legend()
            plt.grid()
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.savefig("RBS_QNN_Accuracy.png", dpi = 300)
            plt.show()

    def SGD_backprop_encoding(self, Input_state, Sample, max_epochs = 1000, eps = 0, learning_rate = 0.5, regularization = 0):
        """ Very Similar to the SGD_backprop function except here we are only
        considering one sample at a time for an encoding purpose. """
        self.Update_Inner_matrices()
        for epoch in range(max_epochs):
            self.feedforward_pass(Sample)
            input_error = self.cost.delta(self.activation.delta(self.Inner_layers[-1]), self.activation.fn(self.Inner_layers[-1]), Sample)
            self.backpropagation(input_error)
            nabla_iteration = self.gradient_derivation()
            self.angles = self.angles - regularization*learning_rate*self.angles -  learning_rate*nabla_iteration
            self.Update_Inner_matrices()



    def ADAM_backprop(self, X_train, Y_train, X_test, Y_test, plot_training = False, nbr_class = 2, batch_size = 5, max_epochs = 50, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8, lbda = 1e-8):
        """ Apply the backpropagation formula and update the list of codes of
        each optimization iteration with ADAM algorithm (to have an optimal
        learning rate). """
        self.Update_Inner_matrices()
        nbr_batches = np.shape(X_train)[0]//batch_size
        List_index_samples = [i for i in range(np.shape(X_train)[0])]
        m, v, t = np.zeros(np.shape(self.angles)), np.zeros(np.shape(self.angles)), 0
        if plot_training:
            Training_acc, Testing_acc = [], []
            _, training_acc = self.cost_accuracy_derivation(X_train, Y_train, nbr_class)
            _, testing_acc = self.cost_accuracy_derivation(X_test, Y_test, nbr_class)
            Training_acc.append(training_acc)
            Testing_acc.append(testing_acc)
        for epoch in tqdm(range(max_epochs)):
            t += 1
            np.random.shuffle(List_index_samples) # We shuffle the samples before each training epoch
            for i in range(nbr_batches):
                gradients = np.zeros(np.shape(self.angles)[0])
                for j in range(batch_size):
                    self.feedforward_pass(X_train[List_index_samples[i*batch_size + j]])
                    input_error = self.cost.delta(self.activation.prime(self.Inner_layers[-1]), self.activation.fn(self.Inner_layers[-1]), Y_train[List_index_samples[i*batch_size + j]])
                    self.backpropagation(input_error)
                    gradients = gradients + self.gradient_derivation()/batch_size
                # Update angles and the list of training cost using ADAM:
                beta_1 = beta_1*(lbda)**(t-1)  # Decay first moment running average coefficient
                m = beta_1*m + (1 - beta_1)*gradients     #Update biased first moment estimate
                v = beta_2*v + (1 - beta_2)*gradients**2  #Update biased second raw moment estimate
                m_hat = m/(1 - beta_1**t)      # Compute bias corrected first moment estimate
                v_hat = v/(1 - beta_2**t)      # Compute bias corrected second raw moment estimate
                self.angles = self.angles - alpha*m_hat/(np.sqrt(v_hat) + eps) # Update parameters
                self.Update_Inner_matrices()
                if plot_training:
                    _, training_acc = self.cost_accuracy_derivation(X_train, Y_train, nbr_class)
                    _, testing_acc = self.cost_accuracy_derivation(X_test, Y_test, nbr_class)
                    Training_acc.append(training_acc)
                    Testing_acc.append(testing_acc)
        if plot_training:
            fig = plt.figure()
            plt.plot([i for i in range(len(Training_acc))], Training_acc, 'b', label = 'Training accuracy')
            plt.plot([i for i in range(len(Testing_acc))], Testing_acc, 'r', label = 'Testing accuracy')
            plt.title('')
            plt.legend()
            plt.grid()
            plt.xlabel('Nbr batches')
            plt.ylabel('Accuracy')
            plt.savefig("RBS_QNN_Accuracy.png", dpi = 300)
            plt.show()



    def ADAM_backprop_encoding(self, Input_state, Sample, max_epochs = 100, alpha = 0.01, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8, lbda = 1e-8):
        """ Very Similar to the ADAM_backprop function except here we are only
        considering one sample at a time for an encoding purpose. """
        self.Update_Inner_matrices()
        m, v, t = np.zeros(np.shape(self.angles)), np.zeros(np.shape(self.angles)), 0
        for epoch in range(max_epochs):
            t += 1
            self.feedforward_pass(Input_state)
            input_error = self.cost.delta(self.activation.delta(self.Inner_layers[-1]), self.activation.fn(self.Inner_layers[-1]), Sample)
            self.backpropagation(input_error)
            gradients = self.gradient_derivation()
            # Update angles and the list of training cost using ADAM:
            beta_1 = beta_1*(lbda)**(t-1)  # Decay first moment running average coefficient
            m = beta_1*m + (1 - beta_1)*gradients     #Update biased first moment estimate
            v = beta_2*v + (1 - beta_2)*gradients**2  #Update biased second raw moment estimate
            m_hat = m/(1 - beta_1**t)      # Compute bias corrected first moment estimate
            v_hat = v/(1 - beta_2**t)      # Compute bias corrected second raw moment estimate
            self.angles = self.angles - alpha*m_hat/(np.sqrt(v_hat) + eps) # Update parameters
            self.Update_Inner_matrices()
