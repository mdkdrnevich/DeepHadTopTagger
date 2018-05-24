
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
parser.add_argument("-s", "--subsets", help="Number of subsets of data to try", default=10, type=int)
args = parser.parse_args()


cuda = th.cuda.is_available()

# ## Load the Datasets
# Here I load the datasets using my custom <code>Dataset</code> class. This ensures that the data is scaled properly and then the PyTorch <code>DataLoader</code> shuffles and iterates over the dataset in batches.

print("Loading Datasets")

trainset = utils.CollisionDataset(args.training)
valset = utils.CollisionDataset(args.validation, scaler=trainset.scaler)

batch_size = args.batch_size

# ## Initialize the NN, Loss Function, and Optimizer

print("Initializing Model")

datasize = trainset.shape[0]
input_dim = trainset.shape[1]

# # Deep Neural Network on the Basic Features

val_curve = []

for i in range(1, args.subsets+1):
    trainset.shuffle()
    valset.shuffle()
    subtrainset = trainset.subsample(i/args.subsets, inplace=False)
    subvalset = valset.subsample(i/args.subsets, inplace=False)
    trainloader = DataLoader(subtrainset, batch_size=batch_size, shuffle=True, num_workers=5)
    validationloader = DataLoader(subvalset, batch_size=batch_size, shuffle=True, num_workers=5)
    num_batches = len(trainloader)
    
    dnet = ConvDHTTNet(input_dim, int(args.width*i/args.subsets))
    if cuda: dnet.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(dnet.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    print("Training DNN {}".format(i))
    
    def cost(model, X, y):
        outputs = model(X)
        return criterion(outputs, y)
    
    for epoch in range(1, args.epochs+1):
        utils.train(dnet, cost, optimizer, trainloader, cuda=cuda)
        losses = utils.test(dnet, cost, trainloader, validationloader, cuda=cuda, scheduler=scheduler)
        
    val_curve.append(losses)
print("Done")

# Plot the training & validation curves for each epoch
fig, ax = plt.subplots()
plt.plot([int((i+1)*datasize/args.subsets) for i in range(len(val_curve))], val_curve)
ax.set_ylabel("BCE Loss")
ax.set_xlabel("Size of Dataset")
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
    fig.savefig("{}_learning_curve.png".format(args.name))
else:
    fig.savefig("learning_curve.png")
