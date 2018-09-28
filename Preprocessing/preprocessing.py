
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

def strToTuple(string):
    return tuple(int(x) for x in string.split('/'))

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--background", help="Directory of background samples (will glob all files)")
parser.add_argument("-s", "--signal", help="Directory of signal samples (will glob all files)")
parser.add_argument("-n", "--name", help="Name of the sample that you want added to the saved datafile names", default="")
parser.add_argument("-t", "--test", action="store_true", help="Save testing sets separately as <file>_test.npy")
parser.add_argument("-v", "--vars", type=int, default=0, help="Integer for which set of variables to use:\n 0 - Basic vars\n 1 - Engineered vars\n 2 - All vars")
parser.add_argument("--split", help="3 numbers separated by '/' of percents for the training/validation/testing split",
                    type=strToTuple, default=(80, 10, 10))
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
       "DeepCSVprobudsg {}", "qgid {}", "ptD {}", "axis1 {}", "mult {}"]]
     for i in range(1, 4)]))

HEADER = RAW_HEADER + list(itertools.chain.from_iterable(
    [[n.format(x) for n in 
      ["{} Pt", "{} Mass", "{} CSV", "{} CvsL", "{} CvsB", "{} ptD", "{} axis1", "{} mult"]]
     for x in ['b', 'Wj1', 'Wj2']]))

HEADER += ["b+Wj1 deltaR", "b+Wj1 Mass", "b+Wj2 deltaR", "b+Wj2 Mass", "W deltaR", "W Mass", "b+W deltaR", "Top Mass"]

#["Top Mass", "Top Pt", "Top ptDR", "W Mass", "W ptDR", "soft drop n2",
#                                   "j2 ptD", "j3 ptD", "(b, j2) mass", "(b, j3) mass"]


if args.vars == 0:
    first_index = 0
    last_index = len(RAW_HEADER) - 1
    datatype = "basic"
elif args.vars == 1:
    first_index = len(RAW_HEADER) - 1
    last_index = len(HEADER) - 1
    datatype = "engineered"
elif args.vars == 2:
    first_index = 0
    last_index = len(HEADER) - 1
    datatype = "all"
else:
    raise ValueError("Invalid input for the -v, --vars option.")


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
# - Training: 80%
# - Validation: 10%
# - Testing: 10%

ix_train_cut = int(args.split[0]*smallest/100)
ix_val_cut = ix_train_cut + int(args.split[1]*smallest/100)
ix_test_cut = ix_val_cut + int(args.split[2]*smallest/100)

sig_scale = len(bkgd_files)/len(signal_files) if len(signal_files) > len(bkgd_files) > 0 else 1
bkgd_scale = len(signal_files)/len(bkgd_files) if len(bkgd_files) > len(signal_files) > 0 else 1

for ix, dset in enumerate(cut_signals):
    dset.subsample(smallest)
    dset.shuffle()
    if args.test:
            name = ospath.splitext(signal_files[ix])[0] + "_test.npy"
            if (ix_test_cut - ix_val_cut) > 0:
                dset.slice(ix_val_cut, ix_test_cut).saveas(name)
    if ix == 0:
        if ix_train_cut > 0:
            total_train_signal = dset.slice(0, int(ix_train_cut * sig_scale))
        if (ix_val_cut - ix_train_cut) > 0:
            total_val_signal = dset.slice(int(ix_train_cut * sig_scale), int(ix_val_cut * sig_scale))
        if (ix_test_cut - ix_val_cut) > 0:
            total_test_signal = dset.slice(int(ix_val_cut * sig_scale), int(ix_test_cut * sig_scale))
    else:
        if ix_train_cut > 0:
            total_train_signal = total_train_signal + dset.slice(0, int(ix_train_cut * sig_scale))
        if (ix_val_cut - ix_train_cut) > 0:
            total_val_signal = total_val_signal + dset.slice(int(ix_train_cut * sig_scale), int(ix_val_cut * sig_scale))
        if (ix_test_cut - ix_val_cut) > 0:
            total_test_signal = total_test_signal + dset.slice(int(ix_val_cut * sig_scale), int(ix_test_cut * sig_scale))
        
for ix, dset in enumerate(cut_bkgds):
    dset.subsample(smallest)
    dset.shuffle()
    if args.test:
            name = ospath.splitext(bkgd_files[ix])[0] + "_test.npy"
            if (ix_test_cut - ix_val_cut) > 0:
                dset.slice(ix_val_cut, ix_test_cut).saveas(name)
    if ix == 0:
        if ix_train_cut > 0:
            total_train_bkgd = dset.slice(0, int(ix_train_cut * bkgd_scale))
        if (ix_val_cut - ix_train_cut) > 0:
            total_val_bkgd = dset.slice(int(ix_train_cut * bkgd_scale), int(ix_val_cut * bkgd_scale))
        if (ix_test_cut - ix_val_cut) > 0:
            total_test_bkgd = dset.slice(int(ix_val_cut * bkgd_scale), int(ix_test_cut * bkgd_scale))
    else:
        if ix_train_cut > 0:
            total_train_bkgd = total_train_bkgd + dset.slice(0, int(ix_train_cut * bkgd_scale))
        if (ix_val_cut - ix_train_cut) > 0:
            total_val_bkgd = total_val_bkgd + dset.slice(int(ix_train_cut * bkgd_scale), int(ix_val_cut * bkgd_scale))
        if (ix_test_cut - ix_val_cut) > 0:
            total_test_bkgd = total_test_bkgd + dset.slice(int(ix_val_cut * bkgd_scale), int(ix_test_cut * bkgd_scale))
        
print("Saving datasets")

if ix_train_cut > 0:
    train = total_train_signal + total_train_bkgd
    train.shuffle()
    train.slice(first_index, last_index, dim=1).saveas(args.name + "training_" + datatype + "_set.npy")
if (ix_val_cut - ix_train_cut) > 0:
    val = total_val_signal + total_val_bkgd
    val.shuffle()
    val.slice(first_index, last_index, dim=1).saveas(args.name + "validation_" + datatype + "_set.npy")
if (ix_test_cut - ix_val_cut) > 0:
    if len(signal_files) > 0 and len(bkgd_files) > 0:
        test = total_test_signal + total_test_bkgd
    elif len(signal_files) > 0:
        test = total_test_signal
    elif len(bkgd_files) > 0:
        test = total_test_bkgd
    test.shuffle()
    test.slice(first_index, last_index, dim=1).saveas(args.name + "testing_" + datatype + "_set.npy")
