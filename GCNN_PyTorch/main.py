# PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import dirname, join as pjoin
import scipy.io as sio
from torch.utils.data import DataLoader


class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


index_subjects = [2, 3, 6, 9, 11, 12, 13, 15, 17, 18, 19, 22, 23, 24, 26]
data_dir_GC = './PythonDatas/GC/'
data_dir_input = './PythonDatas/Input/'

trainingSet = []

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
        current_GC = mat_contents_GC[current_key]
        ready_current_GC = np.array(current_GC)
        current_input = mat_contents_Input[next(iterator_Input)]
        nparray_current_input = np.array(current_input)
        ready_current_input = np.mean(nparray_current_input, axis=-1)
        lista = [ready_current_GC, ready_current_input, current_key]
        trainingSet.append(lista)

trainDataloader = DataLoader(trainingSet, batch_size=2, shuffle=True)

for i, data in enumerate(trainDataloader, 0):
    # print('******************* BATCH ********************* \n')
    lista = data
    print(i)
