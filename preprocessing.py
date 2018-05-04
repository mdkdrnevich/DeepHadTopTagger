
# coding: utf-8

from __future__ import print_function, division
import pandas as pd
import numpy as np
import itertools
import pickle
from random import sample
import utils
import argparse
import glob
import os
import os.path as ospath

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--background", help="Directory of background samples (will glob all files)")
parser.add_argument("-s", "--signal", help="Directory of signal samples (will glob all files)")
parser.add_argument("-n", "--name", help="Name of the sample that you want added to the saved datafile names", default="")
parser.add_argument("-x", "--exclude", action="store_true", help="Exclude the engineered variables")
parser.add_argument("-t", "--test", action="store_true", help="Save testing sets separately as <file>_test.npy")
args = parser.parse_args()

if args.background:
    bkgd_files = glob.glob(args.background + os.sep + "*")
else:
    bkgd_files = glob.glob("./data/background/*")
    
if args.signal:
    signal_files = glob.glob(args.signal + os.sep + "*")
else:
    signal_files = glob.glob("./data/signal/*")

# First I make a header list to name the columns in the dataset. This header will be used for the background as well. Then Pandas is used to read in the data.

RAW_HEADER = ["Class"] + list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}"]]
     for i in range(1, 4)]))

HEADER = RAW_HEADER + ["Top Mass", "Top Pt", "Top ptDR", "W Mass", "W ptDR", "soft drop n2",
                                   "j2 ptD", "j3 ptD", "(b, j2) mass", "(b, j3) mass"]


# Invariant Mass Cut
# Here we take a first look at the signal compared to the background for the invariant mass, with the background scaled down since it is much larger. There is a clear opportunity to cut high mass events and retain high efficiency on the signal. After playing around a little, these cuts were determined to be $99\%$ efficient in the range $m_{top} \in [96, 266]$ GeV. The next plot shows the distribution after making the cuts, unscaled.

smallest = 0

cut_signals = []
for dfile in signal_files:
    print("Loading File: {}".format(dfile))
    data = pd.read_csv(dfile, header=None, names=HEADER)

    # Make cuts at 96 & 266 GeV
    cut_data = data[((data["Top Mass"] > 96) & (data["Top Mass"] < 266))]
    cut_dset = utils.CollisionDataset(cut_data.as_matrix())
    
    data_size = cut_data.shape[0]
    smallest = min([smallest, data_size]) if smallest > 0 else data_size
    cut_signals.append(cut_dset)
    
cut_bkgds = []
for dfile in bkgd_files:
    print("Loading File: {}".format(dfile))
    data = pd.read_csv(dfile, header=None, names=HEADER)

    # Make cuts at 96 & 266 GeV
    cut_data = data[((data["Top Mass"] > 96) & (data["Top Mass"] < 266))]
    cut_dset = utils.CollisionDataset(cut_data.as_matrix())
    
    data_size = cut_data.shape[0]
    smallest = min([smallest, data_size]) if smallest > 0 else data_size
    cut_bkgds.append(cut_dset)
    
print("Finished loading files. The smallest file after Top Mass cuts was {}".format(smallest))

# #### Make the different sets
# Save the Datasets as Training, Validation, and Testing Sets to Facilitate Easy Use in PyTorch and Scikit Learn
# I ensure that there is an equal amount of each class in each set
# Dataset Fractions:
# - Training: 65%
# - Validation: 15%
# - Testing: 20%

ix_train_cut = int(0.6*smallest)
ix_val_cut = ix_train_cut + int(0.15*smallest)

sig_scale = len(bkgd_files)/len(signal_files) if len(signal_files) > len(bkgd_files) else 1
bkgd_scale = len(signal_files)/len(bkgd_files) if len(bkgd_files) > len(signal_files) else 1

for ix, dset in enumerate(cut_signals):
    dset.subsample(smallest)
    dset.shuffle()
    if args.test:
            name = ospath.splitext(signal_files[ix])[0] + "_test.npy"
            dset.slice(ix_val_cut, smallest).saveas(name)
    if ix == 0:
        total_train_signal = dset.slice(0, int(ix_train_cut * sig_scale))
        total_val_signal = dset.slice(int(ix_train_cut * sig_scale), int(ix_val_cut * sig_scale))
        total_test_signal = dset.slice(int(ix_val_cut * sig_scale), int(smallest * sig_scale))
    else:
        total_train_signal = total_train_signal + dset.slice(0, int(ix_train_cut * sig_scale))
        total_val_signal = total_val_signal + dset.slice(int(ix_train_cut * sig_scale), int(ix_val_cut * sig_scale))
        total_test_signal = total_test_signal + dset.slice(int(ix_val_cut * sig_scale), int(smallest * sig_scale))
        
for ix, dset in enumerate(cut_bkgds):
    dset.subsample(smallest)
    dset.shuffle()
    if args.test:
            name = ospath.splitext(bkgd_files[ix])[0] + "_test.npy"
            dset.slice(ix_val_cut, smallest).saveas(name)
    if ix == 0:
        total_train_bkgd = dset.slice(0, int(ix_train_cut * bkgd_scale))
        total_val_bkgd = dset.slice(int(ix_train_cut * bkgd_scale), int(ix_val_cut * bkgd_scale))
        total_test_bkgd = dset.slice(int(ix_val_cut * bkgd_scale), int(smallest * bkgd_scale))
    else:
        total_train_bkgd = total_train_bkgd + dset.slice(0, int(ix_train_cut * bkgd_scale))
        total_val_bkgd = total_val_bkgd + dset.slice(int(ix_train_cut * bkgd_scale), int(ix_val_cut * bkgd_scale))
        total_test_bkgd = total_test_bkgd + dset.slice(int(ix_val_cut * bkgd_scale), int(smallest * bkgd_scale))
        
print("Saving datasets")

train = total_train_signal + total_train_bkgd
val = total_val_signal + total_val_bkgd
test = total_test_signal + total_test_bkgd

train.shuffle()
val.shuffle()
test.shuffle()

if not args.exclude:
    train.saveas(args.name + "training_set.npy")
    val.saveas(args.name + "validation_set.npy")
    test.saveas(args.name + "testing_set.npy")

train.slice(0, len(RAW_HEADER), dim=1).saveas(args.name + "training_basic_set.npy")
val.slice(0, len(RAW_HEADER), dim=1).saveas(args.name + "validation_basic_set.npy")
test.slice(0, len(RAW_HEADER), dim=1).saveas(args.name + "testing_basic_set.npy")