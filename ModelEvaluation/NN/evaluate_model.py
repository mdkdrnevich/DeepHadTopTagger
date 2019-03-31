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
import itertools
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name of the experiment you wish to use for prediction", type=str)
parser.add_argument("datafile", help="Name of the file you wish to use for prediction", type=str)
parser.add_argument("-v", "--vars", type=int, default=0, help="Integer for which set of variables to use:\n 0 - Old basic vars\n 1 - Old basic vars + BDT basic vars\n 2 - Engineered vars\n 3 - All vars")
args = parser.parse_args()

if args.vars > 1:
    from ROOT import TLorentzVector

RAW_HEADER = ["Class"] + list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}", "ptD {}", "axis1 {}", "mult {}"]]
     for i in range(1, 17)]))
JETSIZE = 13

df = pd.read_csv(args.datafile, names=RAW_HEADER, index_col=None)
dataset = df.as_matrix()
for i in xrange(dataset.shape[0]):
    dataset[i, 0] = np.array([int(x) for x in dataset[i,0].split('.')])

params = np.load("{}_standardizer.npz".format(args.name))
mu, sig = (params["mean"].astype("float32"), params["std"].astype("float32"))

if args.vars == 0:
    net = nn_classes.DeepBinClassifier(30, 6, 25).eval().cuda()
elif args.vars == 1:
    net = nn_classes.DeepBinClassifier(39, 6, 25).eval().cuda()
elif args.vars == 2:
    net = nn_classes.DeepBinClassifier(32, 6, 25).eval().cuda()
elif args.vars == 3:
    net = nn_classes.DeepBinClassifier(71, 6, 25).eval().cuda()
else:
    raise ValueError("Invalid input for the -v, --vars option.")
    
net.load_state_dict(th.load("{}_net.pth".format(args.name)))

total = 0
what_to_do = 0
for m in xrange(dataset.shape[0]):
    if (m+1)%10000 == 0: print(m+1)
    line = dataset[m, 1:].astype(np.float32)
    good = dataset[m, 0]
    line = line[~np.isnan(line)]
    if len(line)%JETSIZE != 0: raise Exception("Event data is not a valid shape! It is {}".format(len(line)))
    num_jets = len(line)//JETSIZE
    best_score = 0
    best_triplet = (-1, -1, -1)
    for i in xrange(num_jets - 2):
        for j in xrange(i+1, num_jets - 1):
            for k in xrange(j+1, num_jets):
                triplet = [line[i*JETSIZE:(i+1)*JETSIZE], line[j*JETSIZE:(j+1)*JETSIZE], line[k*JETSIZE:(k+1)*JETSIZE]] # Get three jets
                triplet = list(sorted(triplet, key=lambda x: x[5]+x[6])) # Sort by DeepCSV
                if args.vars == 0:
                    pass # take out the last three variables in each jet
                elif args.vars == 1:
                    pass # Leave as is
                elif args.vars == 2 or args.vars == 3:
                    pass # Create the engineered variables and only keep those
                    eng_triplet = [None]*3
                    eng_triplet[0] = triplet[2]
                    eng_triplet[1:] = list(reversed(sorted(triplet[:2], key=lambda x: x[0])))
                    
                    eng_vars = [None]*4

                    for i in xrange(3):
                        eng_vars[i] = np.array([eng_triplet[i][0], # p_T
                                                eng_triplet[i][3], # mass
                                                eng_triplet[i][5]+eng_triplet[i][6], # DeepCSV
                                                eng_triplet[i][7]/(eng_triplet[i][7] + eng_triplet[i][8]), # CvsL
                                                eng_triplet[i][7]/(eng_triplet[i][7] + eng_triplet[i][5] + eng_triplet[i][6]), # CvsB
                                                eng_triplet[i][10], # ptD
                                                eng_triplet[i][11], # axis1
                                                eng_triplet[i][12]]) # mult
                        
                    tvec1 = TLorentzVector()
                    tvec2 = TLorentzVector()
                    tvec3 = TLorentzVector()
                    tvec1.SetPtEtaPhiM(eng_triplet[0][0], eng_triplet[0][1], eng_triplet[0][2], eng_triplet[0][3])
                    tvec2.SetPtEtaPhiM(eng_triplet[1][0], eng_triplet[1][1], eng_triplet[1][2], eng_triplet[1][3])
                    tvec3.SetPtEtaPhiM(eng_triplet[2][0], eng_triplet[2][1], eng_triplet[2][2], eng_triplet[2][3])
                    W = tvec2 + tvec3
                    top = tvec1 + tvec2 + tvec3
                    
                    temp = abs(tvec1.Phi()-tvec2.Phi())
                    temp = temp if temp <= math.pi else temp - 2*math.pi
                    b_wj1_deltaR = np.sqrt((tvec1.Eta()-tvec2.Eta())**2 + temp**2)
                    b_wj1_mass = (tvec1 + tvec2).M()
                    
                    temp = abs(tvec1.Phi()-tvec3.Phi())
                    temp = temp if temp <= math.pi else temp - 2*math.pi
                    b_wj2_deltaR = np.sqrt((tvec1.Eta()-tvec3.Eta())**2 + temp**2)
                    b_wj2_mass =(*tvec1 + *tvec3).M();
                    
                    temp = abs(tvec2.Phi()-tvec3.Phi())
                    temp = temp if temp <= math.pi else temp - 2*math.pi
                    w_deltaR = np.sqrt((tvec2.Eta()-tvec3.Eta())**2 + temp**2)
                    w_mass = W.M();
                    
                    temp = abs(tvec1.Phi()-W.Phi())
                    temp = temp if temp <= math.pi else temp - 2*math.pi
                    b_w_deltaR = np.sqrt((tvec1.Eta()-W.Eta())**2 + temp**2)
                    top_mass = top.M();
                    
                    eng_vars[3] = np.array([b_wj1_deltaR, b_wj1_mass, b_wj2_deltaR, b_wj2_mass, w_deltaR, w_mass, b_w_deltaR, top_mass])
                    if args.vars == 2:
                        triplet = eng_vars
                    elif args.vars == 3:
                        triplet.extend(eng_vars)
                
                triplet = np.concatenate(triplet) # String into a numpy object
                triplet = (triplet - mu)/sig # Normalize
                triplet = Variable(th.from_numpy(triplet)).view(1, -1).cuda()
                score = net(triplet).cpu().view(1).data.numpy().item()
                if score > best_score:
                    best_score = score
                    best_triplet = (i,j,k)
                elif score == best_score:
                    what_to_do += 1
    total += int((np.array(best_triplet) == good).all()) #If this is the correct triplet
print(what_to_do)
print(total*100/dataset.shape[0])
