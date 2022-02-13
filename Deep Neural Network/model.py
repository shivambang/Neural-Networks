import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self, layerSize) -> None:
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList()
        for inp_size, out_size in zip(layerSize[:-1], layerSize[1:]):
            self.layers.append(nn.Linear(inp_size, out_size))
    
    def forward(self, x):
        for linear in self.layers[:-1]:
            x = linear(x)
            x = nn.ReLU(x)
        linear = self.layers[-1]
        out = linear(x)
        out = nn.Sigmoid(out)
        return out