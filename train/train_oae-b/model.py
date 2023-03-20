import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch 
import pdb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def sample_z(mu):

    m = torch.distributions.RelaxedBernoulli(torch.tensor(0.1).to(device), None, mu)
    
    return m.rsample()
        
class OrderedAutoEncoder_b(nn.Module):
    def __init__(self, hidden_layer_size):
        super().__init__()
        self.size=hidden_layer_size
        
        self.encoder = nn.Sequential(
            nn.Linear(256,self.size, bias=None)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.size,256, bias=None)
        )
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal_(m.weight.data)

    def forward(self, x):
        #encoder
        hidden_layer =self.encoder(x)

        #mask
        p = np.random.randint(hidden_layer.size(1)-1)
        mask = torch.ones_like(hidden_layer, dtype=torch.float)
        mask[:,p+1:] = 0.0
        hidden_layer = hidden_layer * mask

        #sample
        hidden_layer = sample_z(hidden_layer)
        hidden_layer = hidden_layer * mask

        #decoder
        decoded = self.decoder(hidden_layer)
        
        return hidden_layer, decoded