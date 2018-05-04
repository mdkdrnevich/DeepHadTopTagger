
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
import matplotlib.patches as mpatches
from nn_classes import *
import utils
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("training", help="File path to the training set")
parser.add_argument("validation", help="File path to the validation set")
parser.add_argument("-b", "--batch_size", help="Batch size", default=2048, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=10, type=int)
parser.add_argument("-n", "--name", help="Name to help describe the output neural net and standardizer")
args = parser.parse_args()


cuda = th.cuda.is_available()

# ## Load the Datasets
# Here I load the datasets using my custom <code>Dataset</code> class. This ensures that the data is scaled properly and then the PyTorch <code>DataLoader</code> shuffles and iterates over the dataset in batches.

trainset = utils.CollisionDataset(args.training)
valset = utils.CollisionDataset(args.validation, scaler=trainset.scaler)

batch_size = args.batch_size
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=5)
num_batches = len(trainloader)

train_X, train_y = trainset[:]
train_X = Variable(train_X)
#train_y = train_y.numpy()
train_y = Variable(train_y).float().view(-1, 1)

val_X, val_y = valset[:]
val_X = Variable(val_X)
#val_y = val_y.numpy()
val_y = Variable(val_y).float().view(-1, 1)

# ## Initialize the NN, Loss Function, and Optimizer

input_dim = trainset.shape[1]

# # Deep Neural Network on the Basic Features

dnet = DHTTNet(input_dim)
if cuda: dnet.cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(dnet.parameters())

#if th.cuda.device_count() > 1:
#  dnet = nn.DataParallel(dnet)

dnet.eval()
dnet.cpu()
#train_discriminant = dnet(train_X).data.numpy()
#val_discriminant = dnet(val_X).data.numpy()
train_output = dnet(train_X)
val_output = dnet(val_X)

#val_curve = [(roc_auc_score(train_y, train_discriminant), roc_auc_score(val_y, val_discriminant))]
val_curve = [(criterion(train_output, train_y).view(1).data.numpy().item(),
              criterion(val_output, val_y).view(1).data.numpy().item())]

if cuda: dnet.cuda()
dnet.train()

print("Training DNN")
for epoch in range(1, args.epochs+1):
    count = 0
    for batch in trainloader:
        count += 1
        print("Epoch {}: {:.2f}%".format(epoch, round(count*100/num_batches, 2)), end='\r')
        inputs, targets = Variable(batch[0]).float(), Variable(batch[1]).float().view(-1, 1)
        inputs = inputs.view(-1, input_dim) # To fix an error
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = inputs.view(-1, input_dim)
        optimizer.zero_grad()
        
        outputs = dnet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    dnet.cpu()
    dnet.eval()
    #Evaluate the model on the training set
    #train_discriminant = dnet(train_X).data.numpy()
    train_output = dnet(train_X)

    # Evaluate the model on a validation set
    #val_discriminant = dnet(val_X).data.numpy()
    val_output = dnet(val_X)
    
    # Add the ROC AUC to the curve
    #val_curve.append((roc_auc_score(train_y, train_discriminant), roc_auc_score(val_y, val_discriminant)))
    # Add the points to the loss curve
    val_curve.append((criterion(train_output, train_y).view(1).data.numpy().item(),
                      criterion(val_output, val_y).view(1).data.numpy().item()))
    
    if cuda: dnet.cuda()
    dnet.train()
    print()
print("Done")

# Plot the training & validation curves for each epoch
fig, ax = plt.subplots()
plt.plot(range(len(val_curve)), val_curve)
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Epochs Finished")
ax.set_title("Validation Curves")
# Get the default colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Build legend entries
train_patch = mpatches.Patch(color=colors[0], label='Training')
val_patch = mpatches.Patch(color=colors[1], label='Validation')
# Construct the legend
plt.legend(handles=[train_patch, val_patch], loc='lower right')
fig.set_size_inches(18, 10)

dnet.eval()
dnet.cpu()
if args.name:
    fig.savefig("{}_val_curve.png".format(args.name))
    trainset.save_scaler("{}_standardizer.npz".format(args.name))
    th.save(dnet.state_dict(), "{}_net.pth".format(args.name))
else:
    fig.savefig("nn_val_curve.png")
    trainset.save_scaler("data_standardizer.npz")
    th.save(dnet.state_dict(), "neural_net.pth")
