# PyTorch
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import dirname, join as pjoin
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 45

class CustomDataset(Dataset):
    def __init__(self, features, gcs, labels, transform=None, target_transform=None):
        self.labels = labels
        self.features = features
        self.gcs = gcs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        gc = self.gcs[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
            gc = self.transform(gc)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, gc, label


# RETE GCNN

input_size = 18
hidden_sizes = [10, 8]
output_size = 2


class MyGCNNModel(nn.Module):
    def __init__(self, c_in, c_out, number_hidden_layer):
        super().__init__()
        self.hidden_layers = number_hidden_layer
        self.projection = nn.Linear(c_in, c_out)
        self.fl1 = nn.Linear(input_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], output_size)

    def layer_GCNN(self, node_features, adj_matrix):
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_features = self.projection(node_features)
        node_features = torch.bmm(adj_matrix, node_features)
        node_features = node_features / num_neighbours
        return node_features

    def forward(self, node_features, adj_matrix):
        with torch.no_grad():
            for g in range(0, self.hidden_layers):
                node_features = self.layer_GCNN(node_features, adj_matrix)
                node_features = F.relu(node_features)
            node_features = torch.reshape(node_features, (BATCH_SIZE, 18))
        for j in range(0, BATCH_SIZE):
            for i in range(0, 18):
                if math.isnan(node_features[j][i]):
                    node_features[j][i] = 0.0
        x = F.relu(self.fl1(node_features))
        x = F.relu(self.fl2(x))
        output_net = self.fl3(x)
        return output_net


# Preparo il Dataset

index_subjects = [2, 3, 6, 9, 11, 12, 13, 15, 17, 18, 19, 22, 23, 24, 26]
greek_bands = ["alpha_", "beta_", "gamma_", "theta_"]
data_dir_GC = './PythonDatas/GC/'
data_dir_input = './PythonDatas/Input/'

dataset = []

for m in range(len(greek_bands)):
    for i in index_subjects:
        mat_filename = pjoin(data_dir_GC, 'GCmatrix_' + greek_bands[m] + str(i) + '.mat')
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
            ready_current_label = int(ready_current_label)
            current_GC = mat_contents_GC[current_key]
            ready_current_GC = np.array(current_GC)

            # Levo i valori NaN sulla diagonale delle matrici di GC ed eseguo threshold
            for h in range(0, 18):
                ready_current_GC[h, h] = 0.0
                for k in range(0, 18):
                    if ready_current_GC[h][k] > 0.08:
                        ready_current_GC[h][k] = 1.0
                    else:
                        ready_current_GC[h][k] = 0.0
            current_input = mat_contents_Input[next(iterator_Input)]
            nparray_current_input = np.array(current_input)

            # NORMALIZATION
            norm = np.linalg.norm(nparray_current_input, axis=-1)
            for g in range(0, 18):
                 nparray_current_input[g, :] = nparray_current_input[g, :]/norm[g]

            # SCALE in [0,1]
            max = np.amax(nparray_current_input, axis=-1)
            min = np.amin(nparray_current_input, axis=-1)

            for g in range(0, 18):
                nparray_current_input[g, :] = (nparray_current_input[g, :] - min[g]) / (max[g] - min[g])
            ready_current_input = np.mean(nparray_current_input, axis=-1)
            lista = [ready_current_GC, ready_current_input, ready_current_label]
            dataset.append(lista)

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [2880, 360, 360])

train_gc = []
train_label = []
train_input = []

for j in range(len(train_set)):
    train_gc.append(train_set[j][0])
    train_input.append(train_set[j][1])
    train_label.append(train_set[j][2])

trainingSet = CustomDataset(train_input, train_gc, train_label)
trainDataloader = DataLoader(trainingSet, batch_size=BATCH_SIZE, shuffle=True)

val_gc = []
val_label = []
val_input = []

for j in range(len(val_set)):
    val_gc.append(val_set[j][0])
    val_input.append(val_set[j][1])
    val_label.append(val_set[j][2])

validationSet = CustomDataset(val_input, val_gc, val_label)
valDataloader = DataLoader(validationSet, batch_size=BATCH_SIZE, shuffle=True)

test_gc = []
test_label = []
test_input = []

for j in range(len(test_set)):
    test_gc.append(test_set[j][0])
    test_input.append(test_set[j][1])
    test_label.append(test_set[j][2])

testSet = CustomDataset(test_input, test_gc, test_label)
testDataloader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
net = MyGCNNModel(1, 1, 1).double()
optimizer = optim.Adam(net.parameters(), weight_decay=1e-5)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs")

for epoch in range(100):  # loop over the train dataset multiple times
    print('Running Epoch: ', epoch)

    # Epoch Train

    for i, data in enumerate(trainDataloader, 0):
        inputs, gcs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs.double().view(BATCH_SIZE, 18, 1), gcs.double())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    running_loss_train = 0.0

    with torch.no_grad():
        for data in trainDataloader:
            inputs, gcs, labels = data

            outputs = net(inputs.double().view(BATCH_SIZE, 18, 1), gcs.double())
            loss = criterion(outputs, labels)
            running_loss_train += loss.item()

    running_loss_train = running_loss_train / 64

    print("Loss: ", running_loss_train)

    running_loss_valid = 0.0

    with torch.no_grad():
        for data in valDataloader:
            inputs, gcs, labels = data

            outputs = net(inputs.double().view(BATCH_SIZE, 18, 1), gcs.double())
            loss = criterion(outputs, labels)
            running_loss_valid += loss.item()

    running_loss_valid = running_loss_valid / 8

    writer.add_scalars('Loss', {'trainset': running_loss_train, 'validset': running_loss_valid}, epoch + 1)

writer.close()
# tensorboard --logdir=runs
print('Finished Training')

PATH = './last.pth'
torch.save(net.state_dict(), PATH)

net = MyGCNNModel(1, 1, 1).double()
PATH = './last.pth'
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testDataloader:
        inputs, gcs, labels = data
        outputs = net(inputs.double().view(BATCH_SIZE, 18, 1), gcs.double())
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 180 test window: %.2f %%' % (100 * correct / total))
