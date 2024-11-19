import torch
import numpy as np

class fourierNetwork(torch.nn.Module):
    def __init__(self, layers, nonlinearity):
        super(KernelNN, self).__init__()
        
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = torch.nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(torch.nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x