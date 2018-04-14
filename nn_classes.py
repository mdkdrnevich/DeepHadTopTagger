from __future__ import print_function, division
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    xrange
except NameError:
    xrange = range

class CollisionDataset(Dataset):
    def __init__(self, datafile, filetype, header=None, scaler=None, target_col=0, index_col=None):
        if filetype.lower() == "pandas":
            dataframe = pd.read_csv(datafile, header=header, index_col=index_col)
            X = pd.concat([dataframe.iloc[:, :target_col], dataframe.iloc[:, target_col+1:]], axis=1).as_matrix()
            y = dataframe.iloc[:, target_col].as_matrix()
        elif filetype.lower() == "numpy":
            M = np.load(datafile)
            X = np.concatenate([M[:, :target_col], M[:, target_col+1:]], axis=1)
            y = M[:, target_col]
            
        self.scaler = preprocessing.StandardScaler().fit(X) if scaler is None else scaler
        self._tX = th.from_numpy(self.scaler.transform(X)).float()
        self._tY = th.from_numpy(y).type(th.ByteTensor)
            
    
    def __len__(self):
        return len(self._tY)
    
    
    def __getitem__(self, idx):
        return {'input': self._tX[idx].contiguous().view(1, -1) if type(idx) is int else self._tX[idx, :],
                'target': self._tY[idx]}
    
    
    def subsample(self, size):
        datasize = len(self)
        if 0 < size <= 1:
            cut = int(size*datasize)
        elif size > 1:
            cut = min((int(size), datasize))
        else:
            return None
        subsample = random.sample(xrange(datasize), cut)
        self._tX = self._tX[subsample]
        self._tY = self._tY[subsample]
        
        
    def saveas(self, filename, filetype):
        if filetype.lower() == 'numpy':
            np.save(filename, np.concatenate((self._tY.view(-1, 1).numpy(), self.scaler.inverse_transform(self._tX)), axis=1))
            
    
class DHTTNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepBinaryRegNet, self).__init__()
        # Layers
        #  - Linear
        #  - Activation
        #  - Batch Normalization
        #  - Dropout
        self.lin1 = nn.Linear(input_dim, input_dim*2)
        self.f1 = nn.PReLU(input_dim*2)
        self.norm1 = nn.BatchNorm1d(input_dim*2)
        #
        self.lin2 = nn.Linear(input_dim*2, input_dim*5)
        self.f2 = nn.PReLU(input_dim*5)
        self.norm2 = nn.BatchNorm1d(input_dim*5)
        #
        self.lin3 = nn.Linear(input_dim*5, input_dim*10)
        self.f3 = nn.PReLU(input_dim*10)
        self.norm3 = nn.BatchNorm1d(input_dim*10)
        #
        self.lin4 = nn.Linear(input_dim*10, input_dim*15)
        self.f4 = nn.PReLU(input_dim*15)
        self.norm4 = nn.BatchNorm1d(input_dim*15)
        #
        self.lin5 = nn.Linear(input_dim*15, input_dim*10)
        self.f5 = nn.PReLU(input_dim*10)
        self.norm5 = nn.BatchNorm1d(input_dim*10)
        #
        self.lin6 = nn.Linear(input_dim*10, input_dim*5)
        self.f6 = nn.PReLU(input_dim*5)
        self.norm6 = nn.BatchNorm1d(input_dim*5)
        #
        self.lin7 = nn.Linear(input_dim*5, input_dim*2)
        self.f7 = nn.PReLU(input_dim*2)
        self.norm7 = nn.BatchNorm1d(input_dim*2)
        #
        self.lin8 = nn.Linear(input_dim*2, input_dim)
        self.f8 = nn.PReLU(input_dim)
        self.norm8 = nn.BatchNorm1d(input_dim)
        #
        self.lin9 = nn.Linear(input_dim, input_dim//2)
        self.f9 = nn.PReLU(input_dim//2)
        self.norm9 = nn.BatchNorm1d(input_dim//2)
        #
        self.lin10 = nn.Linear(input_dim//2, 1)
        # Dropout Layer
        self.dropout = nn.Dropout(0.2)
        

    def forward(self, x):
        x = self.norm1(self.f1(self.lin1(x)))
        x = self.dropout(self.norm2(self.f2(self.lin2(x))))
        x = self.norm3(self.f3(self.lin3(x)))
        x = self.dropout(self.norm4(self.f4(self.lin4(x))))
        x = self.norm5(self.f5(self.lin5(x)))
        x = self.dropout(self.norm6(self.f6(self.lin6(x))))
        x = self.norm7(self.f7(self.lin7(x)))
        x = self.dropout(self.norm8(self.f8(self.lin8(x))))
        x = self.norm9(self.f9(self.lin9(x)))
        x = F.sigmoid(self.lin10(x))
        return x