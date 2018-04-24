from __future__ import print_function, division
import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import Dataset
from torch.autograd import Variable
import random
from sklearn import preprocessing as skl_preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import os.path as ospath
import itertools

try:
    xrange
except NameError:
    xrange = range
    

class CollisionDataset(Dataset):
    def __init__(self, data, header=None, scaler=None, target_col=0, index_col=None):
        if type(data) is np.ndarray:
            X = np.concatenate([data[:, :target_col], data[:, target_col+1:]], axis=1)
            y = data[:, target_col]
        else:
            filetype = ospath.splitext(data)[1][1:]
            if filetype.lower() == "csv":
                self._dataframe = pd.read_csv(data, header=header, index_col=index_col)
                X = pd.concat([self._dataframe.iloc[:, :target_col], self._dataframe.iloc[:, target_col+1:]], axis=1).as_matrix()
                y = self._dataframe.iloc[:, target_col].as_matrix()
            elif filetype.lower() == "npy":
                M = np.load(data)
                X = np.concatenate([M[:, :target_col], M[:, target_col+1:]], axis=1)
                y = M[:, target_col]
            
        if type(scaler) is skl_preprocessing.data.StandardScaler:
            self.scaler = scaler
            self._scale_type = 'scikit'
        elif hasattr(scaler, '__iter__') and len(scaler) == 2:
            self.scaler = scaler
            self._scale_type = 'manual'
        elif scaler is None:
            self.scaler = skl_preprocessing.StandardScaler().fit(X)
            self._scale_type = 'scikit'
            
        self._tX = th.from_numpy(self._transform(X)).float()
        self._tY = th.from_numpy(y).long().view(-1, 1)
            
    
    def __len__(self):
        return len(self._tY)
    
    
    def __getitem__(self, idx):
        return (th.from_numpy(self._tX.numpy()[idx].reshape((1, -1))) if type(idx) is int else th.from_numpy(self._tX.numpy()[idx]),
                th.from_numpy(self._tY.numpy()[idx[0]]) if type(idx) is tuple else th.from_numpy(self._tY.numpy()[idx]))
    
    
    def __add__(left, right):
        return CollisionDataset(np.concatenate((left._remerge(), right._remerge())))
    
    
    def _transform(self, numpy_array, inverse=False):
        if not inverse:
            if self._scale_type == 'scikit':
                return self.scaler.transform(numpy_array)
            elif self._scale_type == 'manual':
                return (numpy_array - self.scaler[0])/self.scaler[1]
        else:
            if self._scale_type == 'scikit':
                return self.scaler.inverse_transform(numpy_array)
            elif self._scale_type == 'manual':
                return (numpy_array*self.scaler[1]) + self.scaler[0]
        
        
    def _remerge(self):
        return np.concatenate((self._tY.view(-1, 1).numpy(), self._transform(self._tX.numpy(), inverse=True)), axis=1)
    
    
    @property
    def shape(self):
        return self._tX.numpy().shape
    
    
    def subsample(self, size, inplace=True):
        datasize = len(self)
        if 0 < size <= 1:
            cut = int(size*datasize)
        elif size > 1:
            cut = min((int(size), datasize))
        else:
            return None
        subsample = random.sample(xrange(datasize), cut)
        if inplace:
            self._tX = th.from_numpy(self._tX.numpy()[subsample])
            self._tY = th.from_numpy(self._tY.numpy()[subsample])
        else:
            return (th.from_numpy(self._tX.numpy()[subsample]), th.from_numpy(self._tY.numpy()[subsample]))
        
     
    def shuffle(self, inplace=True):
        datasize = len(self)
        indices = list(xrange(datasize))
        random.shuffle(indices)
        if inplace:
            self._tX = th.from_numpy(self._tX.numpy()[indices])
            self._tY = th.from_numpy(self._tY.numpy()[indices])
        else:
            return (th.from_numpy(self._tX.numpy()[indices]), th.from_numpy(self._tY.numpy()[indices]))
        
    
    def slice(self, start, end, dim=0):
        if dim == 0:
            return CollisionDataset(self._remerge()[start:end], scaler=self.scaler)
        elif dim == 1:
            merged = self._remerge()
            return CollisionDataset(np.concatenate((merged[:, 0].reshape(-1, 1), merged[:, start+1:end+1]), axis=1))
    
    
    def saveas(self, filename, filetype=None):
        filetype = ospath.splitext(filename)[1][1:] if not filetype else filetype
        if filetype.lower() == 'npy':
            np.save(filename, self._remerge())
            
            
    def save_scaler(self, filename):
        if filename.endswith(".npz"):
            if self._scale_type == 'scikit':
                np.savez_compressed(filename, mean=self.scaler.mean_.astype('float32'), std=self.scaler.scale_.astype('float32'))
            elif self._scale_type == 'manual':
                np.savez_compressed(filename, mean=self.scaler[0], std=self.scaler[1])
            
            
    def load_scaler(self, filename):
        if filename.endswith(".npz"):
            params = np.load(filename)
            X = self._transform(self._tX.numpy(), inverse=True)
            self._scale_type = 'manual'
            self.scaler = (params["mean"], params["std"])
            self._tX = th.from_numpy(self._transform(X))            


def score(model, dataset, cut=0.5):
    X, y = Variable(dataset[:][0]).float(), dataset[:][1].type(th.ByteTensor).view(-1, 1)
    out = model(X).data
    predicted = (out >= cut).type(th.ByteTensor)
    return (predicted == y).sum()/out.size()[0]


def find_cut(model, dataset, n_steps=100, benchmark="f1"):
    X, y = Variable(dataset[:][0]).float(), dataset[:][1].type(th.ByteTensor).view(-1, 1)
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


def print_stats(discriminant, targets, n_steps=100):
    best_cut = 0
    best_score = -1
    for i in range(n_steps+1):
        cut = i*((discriminant.max() - discriminant.min())/n_steps) + discriminant.min()
        score = f1_score(targets, (discriminant >= cut).astype(np.int32))
        if score > best_score:
            best_score = score
            best_cut = cut
    print(classification_report(targets, (discriminant >= best_cut).astype(np.int32)))
    #print(confusion_matrix(targets, (discriminant >= best_cut).astype(np.int32)))
    print("Area Under Curve: {}".format(roc_auc_score(targets, discriminant)))

    
def overlay_roc_curves(experiments, title=""):
    fig, ax = plt.subplots()
    for exp in experiments:
        roc_points = roc_curve(exp['targets'], exp['discriminant'])
        plt.plot(roc_points[0], roc_points[1], label=exp.get("label"))
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title("ROC Curves {}".format(title))
    plt.legend(loc='lower right')
    fig.set_size_inches(18, 10)