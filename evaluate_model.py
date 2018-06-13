from __future__ import division
import pickle
import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, f1_score
import utils
import nn_classes

RAW_HEADER = ["Class"] + list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}", "ptD {}", "axis1 {}", "mult {}"]]
     for i in range(1, 17)]))

df = pd.read_csv("/afs/crc.nd.edu/user/m/mdrnevic/scratch/ttH_triplets_bdt.csv", names=RAW_HEADER, index_col=None)
dataset = df.as_matrix()
for i in xrange(dataset.shape[0]):
    dataset[i, 0] = np.array([int(x) for x in dataset[i,0].split('.')])

params = np.load("new_vars_standardizer.npz")
mu, sig = (params["mean"].astype("float32"), params["std"].astype("float32"))

net = nn_classes.DeepBinClassifier(33, 4, 20).eval()
net.load_state_dict(th.load("new_vars_net.pth"))

total = 0
what_to_do = 0
for m in xrange(dataset.shape[0]):
    if (m+1)%10000 == 0: print(m+1)
    line = dataset[m, 1:].astype(np.float32)
    good = dataset[m, 0]
    line = line[~np.isnan(line)]
    if len(line)%10 != 0: raise Exception("Event data is not a valid shape! It is {}".format(len(line)))
    num_jets = len(line)//10
    best_score = 0
    best_triplet = (-1, -1, -1)
    for i in xrange(num_jets - 2):
        for j in xrange(i+1, num_jets - 1):
            for k in xrange(j+1, num_jets):
                triplet = [line[i*10:(i+1)*10], line[j*10:(j+1)*10], line[k*10:(k+1)*10]] # Get three jets
                triplet = list(reversed(sorted(triplet, key=lambda x: x[5]+x[6])))
                triplet = np.concatenate(triplet)
                triplet = (triplet - mu)/sig
                triplet = Variable(th.from_numpy(triplet)).view(1, -1)
                score = net(triplet).view(1).data.numpy().item()
                if score > best_score:
                    best_score = score
                    best_triplet = (i,j,k)
                elif score == best_score:
                    what_to_do += 1
    total += int((np.array(best_triplet) == good).all()) #If this is the correct triplet
print(what_to_do)
print(total*100/dataset.shape[0])
