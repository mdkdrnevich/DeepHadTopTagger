from __future__ import print_function, division
import torch as th
import torch.nn as nn
import torch.nn.functional as F

try:
    xrange
except NameError:
    xrange = range
            
    
class DHTTNet(nn.Module):
    def __init__(self, input_dim):
        super(DHTTNet, self).__init__()
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
    
    
class smaller_DHTTNet(DHTTNet):       
    def forward(self, x):
        x = self.norm1(self.f1(self.lin1(x)))
        x = self.dropout(self.norm2(self.f2(self.lin2(x))))
        x = self.norm3(self.f3(self.lin3(x)))
        x = self.dropout(self.norm6(self.f6(self.lin6(x))))
        x = self.norm7(self.f7(self.lin7(x)))
        x = self.dropout(self.norm8(self.f8(self.lin8(x))))
        x = self.norm9(self.f9(self.lin9(x)))
        x = F.sigmoid(self.lin10(x))
        return x
    
    
class smallest_DHTTNet(DHTTNet):        
    def forward(self, x):
        x = self.norm1(self.f1(self.lin1(x)))
        x = self.dropout(self.norm2(self.f2(self.lin2(x))))
        x = self.norm7(self.f7(self.lin7(x)))
        x = self.dropout(self.norm8(self.f8(self.lin8(x))))
        x = self.norm9(self.f9(self.lin9(x)))
        x = F.sigmoid(self.lin10(x))
        return x
    
    
class tiny_DHTTNet(DHTTNet):        
    def forward(self, x):
        x = self.norm1(self.f1(self.lin1(x)))
        x = self.dropout(self.norm8(self.f8(self.lin8(x))))
        x = self.norm9(self.f9(self.lin9(x)))
        x = F.sigmoid(self.lin10(x))
        return x
        