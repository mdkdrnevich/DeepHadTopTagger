
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
parser.add_argument("-b", "--batch-size", help="Batch size", default=2048, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=10, type=int)
parser.add_argument("-n", "--name", help="Name to help describe the output neural net and standardizer")
parser.add_argument("-l", "--learning-rate", help="What the initial learning rate should be", default=1e3, type=float)
parser.add_argument("-w", "--width", help="Width of the hidden layers in the DNN", default=15, type=int)
args = parser.parse_args()


cuda = th.cuda.is_available()

# ## Load the Datasets
# Here I load the datasets using my custom <code>Dataset</code> class. This ensures that the data is scaled properly and then the PyTorch <code>DataLoader</code> shuffles and iterates over the dataset in batches.

print("Loading Datasets")

trainset = utils.CollisionDataset(args.training)
valset = utils.CollisionDataset(args.validation, scaler=trainset.scaler)

batch_size = args.batch_size
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=5)
validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=5)
num_batches = len(trainloader)

# ## Initialize the NN, Loss Function, and Optimizer

print("Initializing Model")

input_dim = trainset.shape[1]

# # Deep Neural Network on the Basic Features

dnet = DHTTNet(input_dim, args.width)
if cuda: dnet.cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(dnet.parameters(), lr=args.learning_rate)
#optimizer = optim.SGD(dnet.parameters(), lr=5e-4, momentum=0.9, nesterov=True)
scheduler = ReduceLROnPlateau(optimizer, verbose=True)

#if th.cuda.device_count() > 1:
#  dnet = nn.DataParallel(dnet)

print("Calculating Initial Loss")

dnet.eval()
val_curve = [(utils.compute_loss(dnet, trainloader, criterion, cuda=cuda),
              utils.compute_loss(dnet, validationloader, criterion, cuda=cuda))]
dnet.train()

print("Training DNN")
for epoch in range(1, args.epochs+1):
    count = 0
    for inputs, targets in trainloader:
        count += 1
        print("Epoch {}: {:.2f}%".format(epoch, round(count*100/num_batches, 2)), end='\r')
        
        inputs, targets = Variable(inputs), Variable(targets)
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        optimizer.zero_grad()
        
        outputs = dnet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Add the points to the loss curve
    dnet.eval()
    train_loss = utils.compute_loss(dnet, trainloader, criterion, cuda=cuda)
    val_loss = utils.compute_loss(dnet, validationloader, criterion, cuda=cuda)
    val_curve.append((train_loss, val_loss))
    dnet.train()
    scheduler.step(val_loss)
    print()
print("Done")

# Plot the training & validation curves for each epoch
fig, ax = plt.subplots()
plt.plot(range(len(val_curve)), val_curve)
ax.set_ylabel("BCE Loss")
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
