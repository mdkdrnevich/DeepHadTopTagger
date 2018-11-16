from __future__ import division, print_function
import pickle
import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, f1_score
import itertools
import argparse

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '{0}..{0}..{0}'.format(os.sep))
import hadTopTools

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name of the experiment you wish to use for prediction", type=str)
parser.add_argument("datafile", help="Name of the file you wish to use for prediction", type=str)
parser.add_argument("outfile", help="Name of the file that you wish to store the predictions in", type=argparse.FileType('w'))
parser.add_argument("-v", "--vars", type=int, default=0, help="Integer for which set of variables to use:\n 0 - Old basic vars\n 1 - Old basic vars + BDT basic vars\n 2 - Engineered vars\n 3 - All vars")
args = parser.parse_args()

RAW_HEADER = ["Class"] + list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}", "ptD {}", "axis1 {}", "mult {}"]]
     for i in range(1, 4)]))

OLD_RAW_HEADER = ["Class"] + list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}"]]
     for i in range(1, 4)]))

ENG_HEADER = list(itertools.chain.from_iterable(
    [[n.format(x) for n in 
      ["{} Pt", "{} Mass", "{} CSV", "{} CvsL", "{} CvsB", "{} ptD", "{} axis1", "{} mult"]]
     for x in ['b', 'Wj1', 'Wj2']])) + ["b+Wj1 deltaR", "b+Wj1 Mass", "b+Wj2 deltaR",
                                        "b+Wj2 Mass", "W deltaR", "W Mass", "b+W deltaR", "Top Mass"]

HEADER = RAW_HEADER + ENG_HEADER

if args.vars == 0:
    DATA_HEADER = OLD_RAW_HEADER
elif args.vars == 1:
    DATA_HEADER = RAW_HEADER
elif args.vars == 2:
    DATA_HEADER = ["Class"] + ENG_HEADER
elif args.vars == 3:
    DATA_HEADER = HEADER
else:
    raise ValueError("Invalid input for the -v, --vars option.")


INPUTSIZE = len(DATA_HEADER) - 1
data = pd.read_csv(args.datafile, header=None, names=HEADER)
data = data[DATA_HEADER].as_matrix()

y_true = data[:, 0]
X = data[:, 1:]

params = np.load("{}_standardizer.npz".format(args.name))
mu, sig = (params["mean"].astype("float32"), params["std"].astype("float32"))
X = (X - mu)/sig

net = hadTopTools.nn.DeepBinClassifier(INPUTSIZE, 6, 25).eval().cuda()
net.load_state_dict(th.load("{}_net.pth".format(args.name)))

for m in xrange(X.shape[0]):
    print("{:.2f}%\r".format(m*100/X.shape[0]), end="")
    varX = Variable(th.from_numpy(X[m])).contiguous().view(1, -1).float().cuda()
    score = net(varX).cpu().view(1).data.numpy().item()
    print("{},{}".format(y_true[m], score), file=args.outfile)
args.outfile.close()