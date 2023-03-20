import torch.nn as nn
import torch.nn.init as init
import numpy as np


class OrderedAutoEncoder(nn.Module):
    def __init__(self, hidden_layer_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256,hidden_layer_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_layer_size, 256)
        )

        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        #encoder
        hidden_layer =self.encoder(x)
        
        #decoder
        decoded = self.decoder(hidden_layer)
        
        return hidden_layer, decoded