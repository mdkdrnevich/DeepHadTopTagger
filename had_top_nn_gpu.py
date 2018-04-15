
# coding: utf-8

# # Neural Network for Hadronic Top Reconstruction
# This file creates a feed-forward binary classification neural network for hadronic top reconstruction by classifying quark jet triplets as being from a top quark or not.

from __future__ import print_function, division
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from nn_classes import *
import utils

# ## Load the Datasets
# Here I load the datasets using my custom <code>Dataset</code> class. This ensures that the data is scaled properly and then the PyTorch <code>DataLoader</code> shuffles and iterates over the dataset in batches.

trainset = utils.CollisionDataset("training_basic_set.npy")
valset = utils.CollisionDataset("validation_basic_set.npy", scaler=trainset.scaler)
testset = utils.CollisionDataset("testing_basic_set.npy", scaler=trainset.scaler)

trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=5)
testloader = DataLoader(testset, batch_size=512, shuffle=True, num_workers=5)

train_X, train_y = trainset[:]
train_X = Variable(train_X)
train_y = train_y.numpy()

val_X, val_y = valset[:]
val_X = Variable(val_X)
val_y = val_y.numpy()

# ## Initialize the NN, Loss Function, and Optimizer

input_dim = trainset.shape[1]

# # Deep Neural Network on the Basic Features

dnet = DHTTNet(input_dim).cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(dnet.parameters())

#if th.cuda.device_count() > 1:
#  dnet = nn.DataParallel(dnet)

dnet.eval()
dnet.cpu()
train_discriminant = dnet(train_X).data.numpy()
val_discriminant = dnet(val_X).data.numpy()
dnet.cuda()
dnet.train()
val_curve = [(roc_auc_score(train_y, train_discriminant), roc_auc_score(val_y, val_discriminant))]

print("Training DNN")
for epoch in range(1, 9):
    if epoch%2 == 0: print(epoch)
    for batch in trainloader:
        inputs, targets = Variable(batch[0].cuda()).float(), Variable(batch[1].cuda()).float().view(-1, 1)
        optimizer.zero_grad()
        
        outputs = dnet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    dnet.cpu()
    dnet.eval()
    #Evaluate the model on the training set
    train_discriminant = dnet(train_X).data.numpy()

    # Evaluate the model on a validation set
    val_discriminant = dnet(val_X).data.numpy()
    dnet.cuda()
    dnet.train()
    
    # Add the ROC AUC to the curve
    val_curve.append((roc_auc_score(train_y, train_discriminant), roc_auc_score(val_y, val_discriminant)))
print("Done")

fig, ax = plt.subplots()
plt.plot(range(len(val_curve)), val_curve)
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Epochs Finished")
ax.set_title("Validation Curves")
handles, _ = ax.get_legend_handles_labels()
labels = ["Training", "Validation"]
plt.legend(handles, labels, loc='lower right')
fig.set_size_inches(18, 10)
fig.savefig("DNN_val_curves.png")

dnet.eval()
dnet.cpu()
th.save(dnet.state_dict(), "neural_net.torch")
