import numpy as np
from scipy.special import binom
import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms

from sklearn.decomposition import PCA
from keras.datasets import mnist, fashion_mnist

from toolbox import map_RBS, RBS_generalized

### MNIST dataset ########################################################
class custom_transform(object):
    def __init__(self, I, map):
        self.I = I
        self.map = map    
    def __call__(self,img):
        img = Image_Basis_B2(self.I, img.resize(self.I, self.I), self.map)
        img = img/torch.linalg.norm(img)
        return(img)    
    def __repr__(self):
        return self.__class__.__name__+'()'

def MNIST_DataLoading(I, batch_size, nbr_class, nbr_train, nbr_test):
    transform = transforms.Compose([transforms.ToTensor(), custom_transform(I, map_RBS(2*I,2))])
    # Train Data
    mnistTrainSet = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    training_set = Subset(mnistTrainSet, range(nbr_train))
    mnistTrainLoader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory=True)
    # Test Data
    mnistTestSet = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_set = Subset(mnistTestSet, range(nbr_test))
    mnistTestLoader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory=True)
    return(mnistTrainLoader, mnistTestLoader)

################################################################################


def Fashion_MNIST(list_class=[i for i in range(10)]):
    """ This function load the Fashion MNIST. We choose the number of class we
    want to consider in this dataset using the list object list_class. """
    print("Loading the Fashion MNIST dataset")
    I, O = 28, 5
    data = fashion_mnist.load_data()
    (X_train_mixed, y_train), (X_test_mixed, y_test) = data
    X_train_mixed, X_test_mixed = X_train_mixed.reshape(60000,I,I), X_test_mixed.reshape(10000, I,I)
    nbr_train, nbr_test = 0, 0
    for y in y_train:
        if (y in list_class):
            nbr_train += 1
    for y in y_test:
        if (y in list_class):
            nbr_test += 1
    X_train, X_test = np.zeros((nbr_train, I, I)), np.zeros((nbr_test, I, I))
    y_vector_train, y_vector_test = np.zeros((nbr_train, int(binom(2*O,2)))), np.zeros((nbr_test, int(binom(2*O,2))))
    new_index = 0
    map = map_RBS(2*O, 2)
    for i, y in enumerate(y_train):
        if (y in list_class):
            X_train[new_index] = X_train_mixed[i]/np.linalg.norm(X_train_mixed[i])
            y_vector_train[new_index][y] = 1
            new_index += 1
    new_index = 0
    for i,y in enumerate(y_test):
        if (y in list_class):
            X_test[new_index] = X_test_mixed[i]/np.linalg.norm(X_test_mixed[i])
            y_vector_test[new_index][y] = 1
            new_index += 1
    return(X_train, y_vector_train, X_test, y_vector_test)


def Fashion_MNIST_pca(list_class, nbr_training_samples, nbr_test_samples, Npca=28**2, I=28, O=int(binom(2*5,2))):
    print("Loading the Fashion MNIST data")
    data = fashion_mnist.load_data()
    (X_train, y_train), (X_test, y_test) = data
    pca = PCA(n_components = Npca)
    X_train, X_test = X_train.reshape(60000, 28**2), X_test.reshape(10000, 28**2)
    X_pca_train, X_pca_test = pca.fit_transform(X_train), pca.fit_transform(X_test)
    var_exp = pca.explained_variance_ratio_
    dictionnary_label = {}
    for i,label in enumerate(list_class):
        dictionnary_label[label] = i
    nbr_train, nbr_test = 0, 0
    for y in y_train:
        if (y in list_class):
            nbr_train += 1
    for y in y_test:
        if (y in list_class):
            nbr_test += 1
    X_train, X_test = np.zeros((nbr_train, Npca)), np.zeros((nbr_test, Npca))
    y_vector_train, y_vector_test = np.zeros((nbr_train, O)), np.zeros((nbr_test, O))
    new_index = 0
    for i, y in enumerate(y_train):
        if (y in list_class):
            X_train[new_index] = X_pca_train[i]/np.linalg.norm(X_pca_train[i])
            y_vector_train[new_index][dictionnary_label[y]] = 1
            new_index += 1
    new_index = 0
    for i,y in enumerate(y_test):
        if (y in list_class):
            X_test[new_index] = X_pca_test[i]/np.linalg.norm(X_pca_test[i])
            y_vector_test[new_index][dictionnary_label[y]] = 1
            new_index += 1
    return(X_train[:nbr_training_samples], y_vector_train[:nbr_training_samples], X_test[:nbr_test_samples], y_vector_test[:nbr_test_samples], var_exp)
