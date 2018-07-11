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
parser.add_argument("-v", "--vars", type=int, default=0, help="Integer for which set of variables to use:\n 0 - Basic vars\n 1 - Engineered vars\n 2 - All vars")
args = parser.parse_args()

RAW_HEADER = ["Class"] + list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}", "ptD {}", "axis1 {}", "mult {}"]]
     for i in range(1, 4)]))

HEADER = RAW_HEADER + list(itertools.chain.from_iterable(
    [[n.format(x) for n in 
      ["{} Pt", "{} Mass", "{} CSV", "{} CvsL", "{} CvsB", "{} ptD", "{} axis1", "{} mult"]]
     for x in ['b', 'Wj1', 'Wj2']]))

HEADER += ["b+Wj1 deltaR", "b+Wj1 Mass", "b+Wj2 deltaR", "b+Wj2 Mass", "W deltaR", "W Mass", "b+W deltaR", "Top Mass"]


INPUTSIZE = len(HEADER) - 1
data = pd.read_csv(args.datafile, header=None, names=HEADER).as_matrix()
if args.vars == 0:
    data = data[:, :len(RAW_HEADER)]
    INPUTSIZE = len(RAW_HEADER) - 1
elif args.vars == 1:
    data = data[:, len(RAW_HEADER):len(HEADER)]
    INPUTSIZE = len(HEADER) - len(RAW_HEADER)

y_true = data[:, 0]
X = data[:, 1:]

params = np.load("{}_standardizer.npz".format(args.name))
mu, sig = (params["mean"].astype("float32"), params["std"].astype("float32"))
X = (X - mu)/sig

net = nn_classes.DeepBinClassifier(INPUTSIZE, 6, 25).eval().cuda()
net.load_state_dict(th.load("{}_net.pth".format(args.name)))

for m in xrange(X.shape[0]):
    print("{:.2f}%\r".format(m*100/X.shape[0]), end="")
    varX = Variable(th.from_numpy(X[m])).view(1, -1).float().cuda()
    score = net(varX).cpu().view(1).data.numpy().item()
    print("{},{}".format(y_true[m], score), file=args.outfile)
args.outfile.close()