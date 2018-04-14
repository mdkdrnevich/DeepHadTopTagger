
# coding: utf-8

# # Neural Network for Hadronic Top Reconstruction
# This file creates a feed-forward binary classification neural network for hadronic top reconstruction by classifying quark jet triplets as being from a top quark or not.

# In[54]:


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
#import torchsample as tsamp
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from nn_classes import *

# In[96]:


def find_cut(model, dataset, n_steps=100, benchmark="f1"):
    X, y = Variable(dataset[:]['input']).float(), dataset[:]['target'].type(th.ByteTensor).view(-1, 1)
    out = model(X).data
    best_cut = 0
    best_score = -1
    for i in range(n_steps+1):
        cut = i*((out.max() - out.min())/n_steps) + out.min()
        if benchmark == "f1":
            score = f1_score(y.numpy(), (out >= cut).type(th.ByteTensor).numpy())
        elif benchmark == "acc":
            score = ((out >= cut).type(th.ByteTensor) == y).sum()
        if score > best_score:
            best_score = score
            best_cut = cut
    return best_cut


# In[89]:


def score(model, dataset, cut=0.5):
    X, y = Variable(dataset[:]['input']).float(), dataset[:]['target'].type(th.ByteTensor).view(-1, 1)
    out = model(X).data
    predicted = (out >= cut).type(th.ByteTensor)
    return (predicted == y).sum()/out.size()[0]


# ## Load the Datasets
# Here I load the datasets using my custom <code>Dataset</code> class. This ensures that the data is scaled properly and then the PyTorch <code>DataLoader</code> shuffles and iterates over the dataset in batches.

# In[18]:


trainset = CollisionDataset("../../scratch/ttH_hadT_cut_train.csv", header=0, target_col=0, index_col=0)
valset = CollisionDataset("../../scratch/ttH_hadT_cut_val.csv", header=0, target_col=0, index_col=0, scaler=trainset.scaler)
testset = CollisionDataset("../../scratch/ttH_hadT_cut_test.csv", header=0, target_col=0, index_col=0, scaler=trainset.scaler)

trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=5)
testloader = DataLoader(testset, batch_size=512, shuffle=True, num_workers=5)

train_X = Variable(trainset[:]['input']).float()
train_y = trainset[:]['target'].long().view(-1, 1).numpy()

val_X = Variable(valset[:]['input']).float()
val_y = valset[:]['target'].long().view(-1, 1).numpy()

# ## Initialize the NN, Loss Function, and Optimizer


#criterion = nn.CrossEntropyLoss()
"""input_dim = trainset._X.shape[1]

net = BinaryNet(input_dim).cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())


# ## Train the Neural Network

# In[6]:

train_discriminant = net(train_X).data.cpu().numpy()
val_discriminant = net(val_X).data.cpu().numpy()
val_curve = [(roc_auc_score(train_y, train_discriminant), roc_auc_score(val_y, val_discriminant))]

print("Training Standard NN")
for epoch in range(1, 11):
    if epoch%2 == 0: print(epoch)
    for batch in trainloader:
        inputs, targets = Variable(batch['input'].cuda()).float(), Variable(batch['target'].cuda()).float().view(-1, 1)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    #Evaluate the model on the training set
    train_discriminant = net(train_X).data.cpu().numpy()

    # Evaluate the model on a validation set
    val_discriminant = net(val_X).data.cpu().numpy()
    
    # Add the ROC AUC to the curve
    val_curve.append((roc_auc_score(train_y, train_discriminant), roc_auc_score(val_y, val_discriminant)))
print("Done")

fig, ax = plt.subplots()
plt.plot(range(1, len(val_curve)+1), val_curve)
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Epochs Finished")
ax.set_title("Validation Curves")
handles, _ = ax.get_legend_handles_labels()
labels = ["Training", "Validation"]
plt.legend(handles, labels, loc='lower right')
fig.set_size_inches(18, 10)
fig.savefig("NN_val_curves.png")

# In[8]:


th.save(net.state_dict(), "neural_net.torch")"""


# # Try Another NN Trained on Just the "Raw" Features

# In[48]:


raw_trainset = CollisionDataset("../../scratch/ttH_hadT_cut_raw_train.csv", header=0, target_col=0, index_col=0)
raw_valset = CollisionDataset("../../scratch/ttH_hadT_cut_raw_val.csv", header=0, target_col=0, index_col=0, scaler=raw_trainset.scaler)
raw_testset = CollisionDataset("../../scratch/ttH_hadT_cut_raw_test.csv", header=0, target_col=0, index_col=0, scaler=raw_trainset.scaler)

raw_trainloader = DataLoader(raw_trainset, batch_size=512, shuffle=True, num_workers=5)
raw_testloader = DataLoader(raw_testset, batch_size=512, shuffle=True, num_workers=5)

raw_train_X = Variable(raw_trainset[:]['input']).float()
raw_train_y = raw_trainset[:]['target'].long().view(-1, 1).numpy()

raw_val_X = Variable(raw_valset[:]['input']).float()
raw_val_y = raw_valset[:]['target'].long().view(-1, 1).numpy()

# In[10]:


raw_input_dim = raw_trainset._X.shape[1]

"""raw_net = BinaryNet(raw_input_dim).cuda()
raw_criterion = nn.BCELoss()
raw_optimizer = optim.Adam(raw_net.parameters())


# In[11]:

train_discriminant = raw_net(raw_train_X).data.cpu().numpy()
val_discriminant = raw_net(raw_val_X).data.cpu().numpy()
val_curve = [(roc_auc_score(raw_train_y, train_discriminant), roc_auc_score(raw_val_y, val_discriminant))]

print("Training Basic NN")
for epoch in range(1, 11):
    if epoch%2 == 0: print(epoch)
    for batch in raw_trainloader:
        inputs, targets = Variable(batch['input'].cuda()).float(), Variable(batch['target'].cuda()).float().view(-1, 1)
        raw_optimizer.zero_grad()
        
        outputs = raw_net(inputs)
        loss = raw_criterion(outputs, targets)
        loss.backward()
        raw_optimizer.step()

    #Evaluate the model on the training set
    train_discriminant = raw_net(raw_train_X).data.cpu().numpy()

    # Evaluate the model on a validation set
    val_discriminant = raw_net(raw_val_X).data.cpu().numpy()
    
    # Add the ROC AUC to the curve
    val_curve.append((roc_auc_score(raw_train_y, train_discriminant), roc_auc_score(raw_val_y, val_discriminant)))
print("Done")

fig, ax = plt.subplots()
plt.plot(range(1, len(val_curve)+1), val_curve)
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Epochs Finished")
ax.set_title("Validation Curves")
handles, _ = ax.get_legend_handles_labels()
labels = ["Training", "Validation"]
plt.legend(handles, labels, loc='lower right')
fig.set_size_inches(18, 10)
fig.savefig("basicNN_val_curves.png")


# In[12]:


th.save(raw_net.state_dict(), "raw_neural_net.torch")"""


# # Deep Neural Network on the Basic Features

# In[13]:


dnet = DeepBinaryRegNet(raw_input_dim).cuda()

deep_criterion = nn.BCELoss()
deep_optimizer = optim.Adam(dnet.parameters())

#if th.cuda.device_count() > 1:
#  dnet = nn.DataParallel(dnet)

# In[14]:

dnet.eval()
dnet.cpu()
train_discriminant = dnet(raw_train_X).data.numpy()
val_discriminant = dnet(raw_val_X).data.numpy()
dnet.cuda()
dnet.train()
val_curve = [(roc_auc_score(raw_train_y, train_discriminant), roc_auc_score(raw_val_y, val_discriminant))]

print("Training DNN")
for epoch in range(1, 9):
    if epoch%2 == 0: print(epoch)
    for batch in raw_trainloader:
        inputs, targets = Variable(batch['input'].cuda()).float(), Variable(batch['target'].cuda()).float().view(-1, 1)
        deep_optimizer.zero_grad()
        
        outputs = dnet(inputs)
        loss = deep_criterion(outputs, targets)
        loss.backward()
        deep_optimizer.step()

    dnet.cpu()
    dnet.eval()
    #Evaluate the model on the training set
    train_discriminant = dnet(raw_train_X).data.numpy()

    # Evaluate the model on a validation set
    val_discriminant = dnet(raw_val_X).data.numpy()
    dnet.cuda()
    dnet.train()
    
    # Add the ROC AUC to the curve
    val_curve.append((roc_auc_score(raw_train_y, train_discriminant), roc_auc_score(raw_val_y, val_discriminant)))
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

# In[16]:

dnet.eval()
dnet.cpu()
th.save(dnet.state_dict(), "deep_basic_neural_net.torch")


# In[97]:


#cut = find_cut(net, trainset, benchmark="acc")
#score(net, valset, cut=cut)
