import torch.nn.functional as F  # functions of the neural network library
import load_dataset as load  # module with function to load data
from Conv_Layer import *
from Pooling import *
import gc
from torch.nn import AdaptiveAvgPool2d
import matplotlib.pyplot as plt
from Dense import *

I = 10
O = I//2
n = 2*I
k = 2
K = 2

device = torch.device("cuda")
batch_size = 1000  # the number of examples per batch
train_loader, test_loader, dim_in, dim_out = load.load_MNIST(batch_size=batch_size)
scala = 100
reduced_loader = reduce_MNIST_dataset(train_loader, scala)
# reduced_loader = filter_dataloader(init_reduced_loader, classes=[0, 1])
number_class = 10
list_gates = [(i, j) for i in range(I-1) for j in range(I-1) if i != j]


full_model = nn.Sequential(Conv_RBS_density_I2(I,K,device), Pooling_2D_density(I,O,device),
                           Conv_RBS_density_I2(I//2,K,device),
                           Basis_Change_I_to_HW_density(O, device), Dense_RBS_density(O, list_gates, number_class, device))

# full_model = nn.Sequential(Basis_Change_I_to_HW_density(I, device), Dense_RBS_density(I, list_gates, number_class, device))

# full_model = nn.Sequential(Conv_RBS_density_I2(I,K,device), Basis_Change_I_to_HW_density(I, device), Dense_RBS_density(I, list_gates, number_class, device))

# loss_function = torch.nn.CrossEntropyLoss()
loss_function = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adagrad(full_model.parameters(), lr=1e-2, lr_decay=1e-6, weight_decay=0, initial_accumulator_value=1e-6, eps=1e-10)
# optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-2)

def train_net(network, train_loader, loss_function, optimizer, device):
    network.train()
    train_loss = 0
    train_accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size()[0] < batch_size: break
        if target.size()[0] < batch_size: break
        # Pooling manually
        new_size = I
        adaptive_avg_pool = AdaptiveAvgPool2d((new_size, new_size))
        data = adaptive_avg_pool(data).to(device)

        init_density_matrix = to_density_matrix(F.normalize(data.squeeze().resize(data.size()[0],I**2), p=2, dim=1).to(device), device)
        # init_density_matrix = to_density_matrix(data.squeeze().resize(data.size()[0],I**2).to(device), device)
        out_network = network(init_density_matrix) # out tensor size: batch * 91 * 91
        # out_network = get_predict_number_vector(out_network, number_class, device)
        # training
        targets = get_batch_projectors(target, batch_size, int(binom(I,2)), number_class, device)
        # targets = get_batch_simple_projectors(target, batch_size, int(binom(2*I,2)), number_class, device)

        loss = loss_function(out_network.to(device), targets.to(device)).to(device) # out_vectors: batch * 10, target: batch * 1
        # loss = loss_function(zero_off_diagonal_elements(out_network.float()),targets.float())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # predict digital number
        predict_number_vector = get_predict_number_vector(out_network, number_class, device)
        predict_number = torch.argmax(predict_number_vector, dim=1).to(device)
        acc = predict_number.eq(target.to(device).view_as(predict_number).to(device)).sum().item()
        train_accuracy += acc

        # delete variable to free memory
        del out_network
        gc.collect()

    train_accuracy /= len(train_loader.dataset)
    train_loss /= (batch_idx + 1)
    return train_accuracy, train_loss


for epoch in range(10):
    train_accuracy, train_loss = train_net(full_model, reduced_loader, loss_function, optimizer, device)
    print(f'Epoch {epoch}: Loss  = {train_loss:.6f}, accuracy = {train_accuracy*100:.4f} %')
#%%
