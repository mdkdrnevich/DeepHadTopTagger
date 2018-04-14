
# coding: utf-8

from __future__ import print_function, division
import pandas as pd
import numpy as np
import itertools
import pickle
from random import sample
import utils

signal_files = ["ttH_hadT_signal.csv"]
bkgd_files = ["ttH_hadT_bkgd.csv"]

# First I make a header list to name the columns in the dataset. This header will be used for the background as well. Then Pandas is used to read in the data.

RAW_HEADER = list(itertools.chain.from_iterable(
    [[n.format(i) for n in 
      ["Pt {}", "Eta {}", "Phi {}", "Mass {}", "Charge {}", "DeepCSVprobb {}", "DeepCSVprobbb {}", "DeepCSVprobc {}",
       "DeepCSVprobudsg {}", "qgid {}"]]
     for i in range(1, 4)]))

HEADER = ["Class"] + RAW_HEADER + ["Top Mass", "Top Pt", "Top ptDR", "W Mass", "W ptDR", "soft drop n2",
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
    
print("Finished loading files")

for ix, dset in enumerate(cut_signals):
    dset.subsample(smallest)
    if ix == 0:
        total_signal = dset
    else:
        total_signal = total_signal + dset
        
for ix, dset in enumerate(cut_bkgds):
    dset.subsample(smallest)
    if ix == 0:
        total_bkgd = dset
    else:
        total_bkgd = total_bkgd + dset

# #### Make the different sets
# Save the Datasets as Training, Validation, and Testing Sets to Facilitate Easy Use in PyTorch and Scikit Learn
# Dataset Fractions:
# - Training: 65%
# - Validation: 15%
# - Testing: 20%

print("Saving datasets")

total_signal.shuffle()
total_bkgd.shuffle()

ix_train_cut = int(0.6*smallest)
ix_val_cut = ix_train_cut + int(0.15*smallest)

train = total_signal.slice(0, ix_train_cut) + total_bkgd.slice(0, ix_train_cut)
val = total_signal.slice(ix_train_cut, ix_val_cut) + total_bkgd.slice(ix_train_cut, ix_val_cut)
test = total_signal.slice(ix_val_cut, smallest) + total_bkgd.slice(ix_val_cut, smallest)

train.saveas("training_set.npy")
val.saveas("validation_set.npy")
test.saveas("testing_set.npy")

train.slice(1, len(RAW_HEADER)+1, dim=1).saveas("training_basic_set.npy")
val.slice(1, len(RAW_HEADER)+1, dim=1).saveas("validation_basic_set.npy")
test.slice(1, len(RAW_HEADER)+1, dim=1).saveas("testing_basic_set.npy")