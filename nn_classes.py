from __future__ import print_function, division
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as rand

try:
    xrange
except NameError:
    xrange = range
            
    
class DHTTNet(nn.Module):
    def __init__(self, input_dim, width):
        super(DHTTNet, self).__init__()
        # Layers
        #  - Linear
        #  - Activation
        #  - Batch Normalization
        #  - Dropout
        self.lin1 = nn.Linear(input_dim, input_dim*width)
        self.f1 = nn.PReLU()
        self.norm1 = nn.BatchNorm1d(input_dim*width)
        #
        self.lin2 = nn.Linear(input_dim*width, input_dim*width)
        self.f2 = nn.PReLU()
        self.norm2 = nn.BatchNorm1d(input_dim*width)
        #
        self.lin3 = nn.Linear(input_dim*width, input_dim*width)
        self.f3 = nn.PReLU()
        self.norm3 = nn.BatchNorm1d(input_dim*width)
        #
        self.lin4 = nn.Linear(input_dim*width, input_dim*width)
        self.f4 = nn.PReLU()
        self.norm4 = nn.BatchNorm1d(input_dim*width)
        #
        self.lin5 = nn.Linear(input_dim*width, input_dim*width)
        self.f5 = nn.PReLU()
        self.norm5 = nn.BatchNorm1d(input_dim*width)
        #
        self.lin6 = nn.Linear(input_dim*width, input_dim*width)
        self.f6 = nn.PReLU()
        self.norm6 = nn.BatchNorm1d(input_dim*width)
        #
        self.lin7 = nn.Linear(input_dim*width, input_dim*width)
        self.f7 = nn.PReLU()
        self.norm7 = nn.BatchNorm1d(input_dim*width)
        #
        self.lin10 = nn.Linear(input_dim*width, 1)
        # Dropout Layer
        self.dropout = nn.Dropout(0.3)
        

    def forward(self, x):
        x = self.norm1(self.f1(self.lin1(x)))
        x = self.dropout(self.norm2(self.f2(self.lin2(x))))
        x = self.norm3(self.f3(self.lin3(x)))
        x = self.dropout(self.norm4(self.f4(self.lin4(x))))
        x = self.norm5(self.f5(self.lin5(x)))
        x = self.dropout(self.norm6(self.f6(self.lin6(x))))
        x = self.norm7(self.f7(self.lin7(x)))
        x = F.sigmoid(self.lin10(x))
        return x

    
class SDAENet(nn.Module):
    def __init__(self, input_dim, num_layers, width):
        super(SDAENet, self).__init__()
        # Layers
        #  - Linear
        #  - Activation
        #  - Batch Normalization
        #  - Dropout
        self.num_layers = num_layers
        self.encoder_dim = input_dim*width
        self.noise_level = 0.3
        self.linear_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        # Add the first hidden layer
        self.linear_layers.append(nn.Linear(input_dim, input_dim*width))
        self.activation_layers.append(nn.PReLU())
        self.norm_layers.append(nn.BatchNorm1d(input_dim*width))
        # Add the rest
        for i in range(num_layers-1):
            self.linear_layers.append(nn.Linear(input_dim*width, input_dim*width))
            self.activation_layers.append(nn.PReLU())
            self.norm_layers.append(nn.BatchNorm1d(input_dim*width))
        if num_layers == 1:
            self.output_layer = nn.Linear(input_dim*width, input_dim)
        else:
            self.output_layer = nn.Linear(input_dim*width, input_dim*width)

        
        
    def noisy(self, x):
        x.requires_grad = False
        features = x.data.shape[1]
        choice = th.from_numpy(rand.choice(np.arange(features), int(features*self.noise_level), replace=False).astype('int32')).long()
        x.index_fill_(1, choice, 0.0)
        x.requires_grad = True
        return x
        

    def forward(self, x):
        x.reguires_grad = False
        # Get the features into the last layer
        N = self.num_layers - 1
        for i in range(N):
            x = self.norm_layers[i](self.activation_layers[i](self.linear_layers[i](x)))
        x.requires_grad = True
        h = x.clone()
        x = self.noisy(x)
        x = self.norm_layers[N](self.activation_layers[N](self.linear_layers[N](x)))
        return (self.output_layer(x), h)
    
    
    def __freezer(self, num, value):
        for p in self.linear_layers[num].parameters():
            p.requires_grad = value
        for p in self.activation_layers[num].parameters():
            p.requires_grad = value
        for p in self.norm_layers[num].parameters():
            p.requires_grad = value
    
    
    def freeze(self, layer_num):
        if hasattr(layer_num, '__iter__'):
            for i in layer_num:
                self.__freezer(i, False)
        else:
            self.__freezer(layer_num, False)
    
    
    def unfreeze(self, layer_num):
        if hasattr(layer_num, '__iter__'):
            for i in layer_num:
                self.__freezer(i, True)
        else:
            self.__freezer(layer_num, True)
    
    
    def add_layer(self):
        w = self.encoder_dim
        self.num_layers += 1
        self.linear_layers.append(nn.Linear(w, w))
        self.activation_layers.append(nn.PReLU())
        self.norm_layers.append(nn.BatchNorm1d(w))
        self.output_layer = nn.Linear(w, w)
    
    
    def get_encoder(self):
        return (self.linear_layers, self.activation_layers, self.norm_layers)


    def grad_parameters(self):
        return (p for p in self.parameters() if p.requires_grad is True)
    
    
class FineTuneNet(nn.Module):
    def __init__(self, encoder):
        super(FineTuneNet, self).__init__()
        for p in encoder.parameters():
            p.requires_grad = True
        layers = encoder.get_encoder()
        self.layers = nn.ModuleList()
        for i in range(encoder.num_layers):
            for j in range(len(layers)):
                self.layers.append(layers[j][i])
        self.output_layer = nn.Linear(encoder.encoder_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if (i+1)%3 == 0 and ((i+1)//3)%2 == 0:
                x = self.dropout(x)
        return F.sigmoid(self.output_layer(x))
    
    
class ConvDHTTNet(nn.Module):
    def __init__(self, input_dim, width):
        super(ConvDHTTNet, self).__init__()
        self.input_dim = input_dim
        self.deep_input_dim = input_dim + 2*1*4 + 1*1*4
        # Layers
        #  - Linear
        #  - Activation
        #  - Batch Normalization
        #  - Dropout
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 4, (2, 5))
        self.conv2 = nn.Conv2d(1, 4, (3, 5))
        self.convf1 = nn.PReLU(4)
        self.convf2 = nn.PReLU(4)
        self.convnorm1 = nn.BatchNorm2d(4)
        self.convnorm2 = nn.BatchNorm2d(4)
        #
        self.lin1 = nn.Linear(self.deep_input_dim, self.deep_input_dim*width)
        self.f1 = nn.PReLU()
        self.norm1 = nn.BatchNorm1d(self.deep_input_dim*width)
        #
        self.lin2 = nn.Linear(self.deep_input_dim*width, self.deep_input_dim*width)
        self.f2 = nn.PReLU()
        self.norm2 = nn.BatchNorm1d(self.deep_input_dim*width)
        #
        self.lin3 = nn.Linear(self.deep_input_dim*width, self.deep_input_dim*width)
        self.f3 = nn.PReLU()
        self.norm3 = nn.BatchNorm1d(self.deep_input_dim*width)
        #
        self.lin4 = nn.Linear(self.deep_input_dim*width, 1)
        # Dropout Layer
        self.dropout = nn.Dropout(0.3)
        

    def forward(self, x):
        # Apply the Convolutional layers to the Pt, Eta, Phi, Mass, Charge values
        batch_size = len(x)
        num_vars = self.input_dim//3
        jet1 = x[:, :5].contiguous().view(batch_size, 1, 1, 5)
        jet2 = x[:, num_vars:num_vars+5].contiguous().view(batch_size, 1, 1, 5)
        jet3 = x[:, num_vars*2:num_vars*2 + 5].contiguous().view(batch_size, 1, 1, 5)
        
        convx = th.cat((jet1, jet2, jet3), dim=2)
        feature1 = self.convnorm1(self.convf1(self.conv1(convx))).view(batch_size, -1)
        feature2 = self.convnorm2(self.convf2(self.conv2(convx))).view(batch_size, -1)
        
        x = th.cat((x, feature1, feature2), dim=1)
        x = self.norm1(self.f1(self.lin1(x)))
        x = self.dropout(self.norm2(self.f2(self.lin2(x))))
        x = self.norm3(self.f3(self.lin3(x)))
        x = F.sigmoid(self.lin4(x))
        return x
