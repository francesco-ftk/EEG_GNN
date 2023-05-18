# PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import dirname, join as pjoin
import scipy.io as sio
from torch.utils.data import DataLoader


class MyGCNNModel(nn.Module):
    def __init__(self, c_in, c_out, number_hidden_layer):
        super().__init__()
        self.hidden_layers = number_hidden_layer
        self.projection = nn.Linear(c_in, c_out)

    def layer_GCNN(self, node_features, adj_matrix):
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_features = self.projection(node_features)
        node_features = torch.bmm(adj_matrix, node_features)
        node_features = node_features / num_neighbours
        return node_features

    def forward(self, node_features, adj_matrix):
        for g in range(0, self.hidden_layers):
            node_features = self.layer_GCNN(node_features, adj_matrix)
            # node_features = F.relu(node_features)
        return node_features


input_size = 18
hidden_sizes = [10]
output_size = 2


class NetMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fl1 = nn.Linear(input_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], output_size)

    def forward(self, x):
        x = F.relu(self.fl1(x))
        x = self.fl2(x)
        return x


# TODO:
#  La rete GCNN serve solo per creare una rappresentazione del segnale
#  La rete Lineare in fondo invece è quella da addestrare presa la rappresentazione a imparare a predire
#  Guardare Graph Attenction Capitol


# Preparo il Dataset

index_subjects = [2, 3, 6, 9, 11, 12, 13, 15, 17, 18, 19, 22, 23, 24, 26]
data_dir_GC = './PythonDatas/GC/'
data_dir_input = './PythonDatas/Input/'

dataset = []

for i in index_subjects:
    mat_filename = pjoin(data_dir_GC, 'GCmatrix_beta_' + str(i) + '.mat')
    mat_contents_GC = sio.loadmat(mat_filename)
    mat_filename = pjoin(data_dir_input, 'Input_beta_' + str(i) + '.mat')
    mat_contents_Input = sio.loadmat(mat_filename)
    iterator_GC = iter(mat_contents_GC)
    iterator_Input = iter(mat_contents_Input)
    for k in range(0, 3, 1):
        next(iterator_GC)
        next(iterator_Input)
    for j in range(3, len(mat_contents_Input.keys())):
        current_key = next(iterator_GC)
        ready_current_label = current_key[-1:]
        current_GC = mat_contents_GC[current_key]
        ready_current_GC = np.array(current_GC)
        # Levo i valori NaN sulla diagonale delle matrici di GC
        for h in range(0, 18):
            ready_current_GC[h, h] = 0.0
        current_input = mat_contents_Input[next(iterator_Input)]
        nparray_current_input = np.array(current_input)
        ready_current_input = np.mean(nparray_current_input, axis=-1)
        lista = [ready_current_GC, ready_current_input, ready_current_label]
        dataset.append(lista)

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [720, 90, 90])
trainDataloader = DataLoader(train_set, batch_size=2, shuffle=True)

for i, data in enumerate(trainDataloader, 0):
    lista = data
    input_data = lista[1][0]
    GC = lista[0][0]

data = []
for i in range(0, len(input_data)):
    data.append([input_data[i]])
b = torch.FloatTensor(data).view(1, 18, 1)

for j in range(0, 18):
    for i in range(0, 18):
        if GC[i][j] > 0.1:
            GC[i][j] = 1.0
        else:
            GC[i][j] = 0.0

a = GC.tolist()
GC = torch.FloatTensor(a).view(1, 18, 18)

print("b:", b)
net = MyGCNNModel(1, 1, 1)
output = net(b, GC)
print("Output: ", output)  # TODO relu???

# prediction = linearNet(output)
# print(prediction) TODO ??? riconvertire a lista di 18 elementi

#  TODO
#   1) Modificare struttura dati passati (Non da fare)
#   2) se Matrici di GC float value portarli in binario (Provare?)
