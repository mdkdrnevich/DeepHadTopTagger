from __future__ import print_function, division
import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import Dataset
import random
from sklearn import preprocessing
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
            
        self.scaler = preprocessing.StandardScaler().fit(X) if scaler is None else scaler
        self._tX = th.from_numpy(self.scaler.transform(X)).float()
        self._tY = th.from_numpy(y).long().view(-1, 1)
            
    
    def __len__(self):
        return len(self._tY)
    
    
    def __getitem__(self, idx):
        return (th.from_numpy(self._tX.numpy()[idx].reshape((1, -1))) if type(idx) is int else th.from_numpy(self._tX.numpy()[idx]),
                th.from_numpy(self._tY.numpy()[idx[0]]) if type(idx) is tuple else th.from_numpy(self._tY.numpy()[idx]))
    
    
    def __add__(left, right):
        return CollisionDataset(np.concatenate((left._remerge(), right._remerge())))
        
        
    def _remerge(self):
        return np.concatenate((self._tY.view(-1, 1).numpy(), self.scaler.inverse_transform(self._tX.numpy())), axis=1)
    
    
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


def score(model, dataset, cut=0.5):
    X, y = Variable(dataset[:]['input']).float(), dataset[:]['target'].type(th.ByteTensor).view(-1, 1)
    out = model(X).data
    predicted = (out >= cut).type(th.ByteTensor)
    return (predicted == y).sum()/out.size()[0]


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