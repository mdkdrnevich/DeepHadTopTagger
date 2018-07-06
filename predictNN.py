from __future__ import division, print_function
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

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name of the experiment you wish to use for prediction", type=str)
parser.add_argument("datafile", help="Name of the file you wish to use for prediction", type=str)
parser.add_argument("outfile", help="Name of the file that you wish to store the predictions in", type=argparse.FileType('w'))
args = parser.parse_args()

RAW_HEADER = ["Class"] + list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}", "ptD {}", "axis1 {}", "mult {}"]]
     for i in range(1, 4)]))

HEADER = RAW_HEADER + ["Top Mass", "Top Pt", "Top ptDR", "W Mass", "W ptDR", "soft drop n2",
                                   "j2 ptD", "j3 ptD", "(b, j2) mass", "(b, j3) mass"]
JETSIZE = 13

data = pd.read_csv(args.datafile, header=None, names=HEADER).as_matrix()
data = data[:, :len(RAW_HEADER)]
y_true = data[:, 0]
X = data[:, 1:]

params = np.load("{}_standardizer.npz".format(args.name))
mu, sig = (params["mean"].astype("float32"), params["std"].astype("float32"))

net = nn_classes.DeepBinClassifier(39, 6, 25).eval().cuda()
net.load_state_dict(th.load("{}_net.pth".format(args.name)))

for m in xrange(X.shape[0]):
    print("{:.2f}%\r".format(m*100/X.shape[0]), end="")
    triplet = [X[m, :JETSIZE], X[m, JETSIZE:2*JETSIZE], X[m, 2*JETSIZE:]]
    triplet = list(sorted(triplet, key=lambda x: x[5]+x[6]))
    triplet = np.concatenate(triplet)
    triplet = (triplet - mu)/sig
    triplet = Variable(th.from_numpy(triplet)).view(1, -1).float().cuda()
    score = net(triplet).cpu().view(1).data.numpy().item()
    print("{},{}".format(y_true[m], score), file=args.outfile)
args.outfile.close()