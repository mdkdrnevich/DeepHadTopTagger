
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
import nn_classes
import utils
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("training", help="File path to the training set")
parser.add_argument("validation", help="File path to the validation set")
parser.add_argument("-b", "--batch-size", help="Batch size", default=2048, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=10, type=int)
parser.add_argument("-n", "--name", help="Name to help describe the output neural net and standardizer")
parser.add_argument("-a", "--learning-rate", help="What the initial learning rate should be", default=1e3, type=float)
parser.add_argument("-l", "--layers", help="Number of hidden layers in the DNN", default=3, type=int)
parser.add_argument("-w", "--width", help="Width of the hidden layers in the DNN", default=15, type=int)
args = parser.parse_args()


cuda = th.cuda.is_available()

# ## Load the Datasets
# Here I load the datasets using my custom <code>Dataset</code> class. This ensures that the data is scaled properly and then the PyTorch <code>DataLoader</code> shuffles and iterates over the dataset in batches.

print("Loading Datasets")

trainset = utils.CollisionDataset(args.training)
valset = utils.CollisionDataset(args.validation, scaler=trainset.scaler)
trainsetAE = utils.AutoencoderDataset(args.training, target_col=0)
valsetAE = utils.AutoencoderDataset(args.validation, target_col=0, scaler=trainset.scaler)

batch_size = args.batch_size
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=5)
validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=5)
trainloaderAE = DataLoader(trainsetAE, batch_size=batch_size, shuffle=True, num_workers=5)
validationloaderAE = DataLoader(valsetAE, batch_size=batch_size, shuffle=True, num_workers=5)
num_batches = len(trainloader)

# ## Initialize the NN, Loss Function, and Optimizer

print("Initializing SDAE Model")

input_dim = trainset.shape[1]

# # Stacked Denoising Autoencoder on the Basic Features

anet = nn_classes.SDAENet(input_dim, 1, args.width)
if cuda: anet.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(anet.parameters(), lr=args.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, verbose=True)
training_params = {'cuda': cuda,
                   'sdae': True,
                   'scheduler': scheduler}

# Define the way to compute the loss and return it
# Add a penalty term that makes the last hidden layer activations sparse
def costSDAE(model, X, y, p=0.05, beta=0.4):
    outputs, targets = model(X)
    pmat = Variable(th.log(th.ones(model.activations.size())*p))
    if cuda:
        pmat = pmat.cuda()
    return criterion(outputs, targets) + beta*F.kl_div(pmat, th.log(model.activations))

#if th.cuda.device_count() > 1:
#    dnet = nn.DataParallel(dnet)

print("Calculating Initial Loss for the Autoencoder")

val_curveAE = [utils.test(anet, costSDAE, trainloaderAE, validationloaderAE, **training_params)]

print("Training the SDAE")

current_num_layers = 1
for epoch in range(1, 21):
    utils.train(anet, costSDAE, optimizer, trainloaderAE, **training_params)
    losses = utils.test(anet, costSDAE, trainloaderAE, validationloaderAE, **training_params)
    val_curveAE.append(losses)

while current_num_layers < args.layers:
    # Add another layer to the autoencoder with dimension 1.5 times the size of its input
    anet.add_layer(2)
    anet.freeze(range(current_num_layers))
    if cuda: anet.cuda()
    optimizer = optim.Adam(anet.grad_parameters(), lr=args.learning_rate)
    training_params['scheduler'] = ReduceLROnPlateau(optimizer, verbose=True)
    current_num_layers += 1
    # Calculate the initial error
    losses = utils.test(anet, costSDAE, trainloaderAE, validationloaderAE, **training_params)
    val_curveAE.append(losses)
    # Re-train the AE with the previous layers frozen
    for epoch in range(1, 21):
        utils.train(anet, costSDAE, optimizer, trainloaderAE, **training_params)
        losses = utils.test(anet, costSDAE, trainloaderAE, validationloaderAE, **training_params)
        val_curveAE.append(losses)

print("Finished Training the SDAE")

anet.eval()
anet.cpu()
if args.name:
    th.save(anet.state_dict(), "{}_sdae_net.pth".format(args.name))
else:
    th.save(anet.state_dict(), "{}_sdae_net.pth".format(args.name))

print("Initializing the Fine Tuning Model")

dnet = nn_classes.FineTuneNet(anet)
if cuda: dnet.cuda()
    
criterion = nn.BCELoss()
optimizer = optim.Adam(dnet.parameters(), lr=args.learning_rate)
#optimizer = optim.SGD(dnet.parameters(), lr=5e-4, momentum=0.9, nesterov=True)
#scheduler = ReduceLROnPlateau(optimizer, verbose=True)
scheduler = None
training_params = {'cuda': cuda,
                   'sdae': False,
                   'scheduler': scheduler}

# Define the way to compute the loss and return it
def cost(model, X, y):
    outputs = model(X)
    return criterion(outputs, y)
    
print("Calculating Initial Loss for the Fine Tuning")

val_curve = [utils.test(dnet, cost, trainloader, validationloader, **training_params)]

print("Fine Tuning the NN")
for epoch in range(1, args.epochs+1):
    utils.train(dnet, cost, optimizer, trainloader, **training_params)
    losses = utils.test(dnet, cost, trainloader, validationloader, **training_params)
    val_curve.append(losses)
print("Done")

autoencoder_fig = utils.plot_curves(val_curveAE, title='Autoencoder Loss Curves')
fine_tune_fig = utils.plot_curves(val_curve, title='Fine Tuning Loss Curves')

dnet.eval()
dnet.cpu()
if args.name:
    fine_tune_fig.savefig("{}_val_curve_fine_tune.png".format(args.name))
    autoencoder_fig.savefig("{}_val_curveAE.png".format(args.name))
    trainset.save_scaler("{}_standardizer.npz".format(args.name))
    th.save(dnet.state_dict(), "{}_fine_tuned_net.pth".format(args.name))
else:
    fine_tune_fig.savefig("{}_val_curve_fine_tune.png".format(args.name))
    autoencoder_fig.savefig("{}_val_curveAE.png".format(args.name))
    trainset.save_scaler("data_standardizer.npz")
    th.save(dnet.state_dict(), "{}_fine_tuned_net.pth".format(args.name))
