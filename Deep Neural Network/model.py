import torch
from torch import tensor
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self, layerSize, p=0) -> None:
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList()
        self.drop = nn.Dropout(p=p)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        for inp_size, out_size in zip(layerSize[:-1], layerSize[1:]):
            self.layers.append(nn.Linear(inp_size, out_size))
    
    def forward(self, x):
        for linear in self.layers[:-1]:
            x = linear(x)
            x = self.relu(x)
            x = self.drop(x)
        linear = self.layers[-1]
        out = linear(x)
        out = self.sigm(out)
        return out