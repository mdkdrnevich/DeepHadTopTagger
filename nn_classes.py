from __future__ import print_function, division
import pandas as pd
from sklearn import preprocessing
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CollisionDataset(Dataset):
    def __init__(self, datafile, header=None, scaler=None, target_col=0, index_col=None):
        self.dataframe = pd.read_csv(datafile, header=header, index_col=index_col)
        self._X = pd.concat([self.dataframe.iloc[:, :target_col], self.dataframe.iloc[:, target_col+1:]], axis=1)
        self._y = self.dataframe.iloc[:, target_col]
        self.scaler = preprocessing.StandardScaler().fit(self._X) if scaler is None else scaler
        self._tX = th.from_numpy(self.scaler.transform(self._X.as_matrix()))
        self._tY = th.from_numpy(self._y.as_matrix())
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if type(idx) is int:
            return {'input': self._tX[idx, :].view(1, -1),
                    'target': self._tY[idx]}
            #return {'input': th.from_numpy(self.scaler.transform(self._X.iloc[idx, :].as_matrix().reshape(1, -1)).flatten()),
            #        'target': self._y.iloc[idx]}
        #return {'input': th.from_numpy(self.scaler.transform(self._X.iloc[idx, :].as_matrix())),
        #        'target': th.from_numpy(self._y.iloc[idx].as_matrix())}
        return {'input': self._tX[idx, :], 'target': self._tY[idx]}
    

# Basic Forward Prop ReLu 
class MultiNet(nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()
        self.lin1 = nn.Linear(3, 7)
        self.lin2 = nn.Linear(7, 3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.softmax(self.lin2(x), dim=1)
        return x
    
class BinaryNet(nn.Module):
    def __init__(self, input_dim):
        super(BinaryNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, (input_dim+1)//2)
        self.lin2 = nn.Linear((input_dim+1)//2, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.sigmoid(self.lin2(x))
        return x
    
class DeepBinaryNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepBinaryNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim*2)
        self.lin2 = nn.Linear(input_dim*2, input_dim*5)
        self.lin3 = nn.Linear(input_dim*5, input_dim*10)
        self.lin4 = nn.Linear(input_dim*10, input_dim*15)
        self.lin5 = nn.Linear(input_dim*15, input_dim*10)
        self.lin6 = nn.Linear(input_dim*10, input_dim*5)
        self.lin7 = nn.Linear(input_dim*5, input_dim*2)
        self.lin8 = nn.Linear(input_dim*2, input_dim)
        self.lin9 = nn.Linear(input_dim, input_dim//2)
        self.lin10 = nn.Linear(input_dim//2, 1)


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.lin6(x))
        x = F.relu(self.lin7(x))
        x = F.relu(self.lin8(x))
        x = F.relu(self.lin9(x))
        x = F.sigmoid(self.lin10(x))
        return x
    
class DeepBinaryDropNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepBinaryDropNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim*2)
        self.lin2 = nn.Linear(input_dim*2, input_dim*5)
        self.lin3 = nn.Linear(input_dim*5, input_dim*10)
        self.lin4 = nn.Linear(input_dim*10, input_dim*15)
        self.lin5 = nn.Linear(input_dim*15, input_dim*10)
        self.lin6 = nn.Linear(input_dim*10, input_dim*5)
        self.lin7 = nn.Linear(input_dim*5, input_dim*2)
        self.lin8 = nn.Linear(input_dim*2, input_dim)
        self.lin9 = nn.Linear(input_dim, input_dim//2)
        self.lin10 = nn.Linear(input_dim//2, 1)
        self.dropout = nn.Dropout()


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.dropout(self.lin3(x)))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.dropout(self.lin6(x)))
        x = F.relu(self.lin7(x))
        x = F.relu(self.lin8(x))
        x = F.relu(self.dropout(self.lin9(x)))
        x = F.sigmoid(self.lin10(x))
        return x
    
class DeepFunnelNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepFunnelNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, 25)
        self.lin2 = nn.Linear(25, 25)
        self.lin3 = nn.Linear(25, 20)
        self.lin4 = nn.Linear(20, 20)
        self.lin5 = nn.Linear(20, 15)
        self.lin6 = nn.Linear(15, 15)
        self.lin7 = nn.Linear(15, 10)
        self.lin8 = nn.Linear(10, 10)
        self.lin9 = nn.Linear(10, 5)
        self.lin10 = nn.Linear(5, 5)
        self.lin11 = nn.Linear(5, 1)
        self.dropout = nn.Dropout()


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.dropout(self.lin3(x)))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.dropout(self.lin6(x)))
        x = F.relu(self.lin7(x))
        x = F.relu(self.lin8(x))
        x = F.relu(self.dropout(self.lin9(x)))
        x = F.relu(self.lin10(x))
        x = F.sigmoid(self.lin11(x))
        return x
    
class DeepConstantNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepConstantNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim*3//2)
        self.lin2 = nn.Linear(input_dim*3//2, input_dim*3//2)
        self.lin3 = nn.Linear(input_dim*3//2, input_dim*3//2)
        self.lin4 = nn.Linear(input_dim*3//2, input_dim*3//2)
        self.lin5 = nn.Linear(input_dim*3//2, input_dim*3//2)
        self.lin6 = nn.Linear(input_dim*3//2, input_dim*3//2)
        self.lin7 = nn.Linear(input_dim*3//2, input_dim*3//2)
        self.lin8 = nn.Linear(input_dim*3//2, input_dim)
        self.lin9 = nn.Linear(input_dim, input_dim//2)
        self.lin10 = nn.Linear(input_dim//2, 1)
        self.dropout = nn.Dropout()


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.dropout(self.lin3(x)))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.dropout(self.lin6(x)))
        x = F.relu(self.lin7(x))
        x = F.relu(self.lin8(x))
        x = F.relu(self.dropout(self.lin9(x)))
        x = F.sigmoid(self.lin10(x))
        return x
    
class DeepBinaryRegNet(nn.Module):
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