import numpy as np
from torch import nn
from torch.nn import functional as F

class CIFARencoder(nn.Module):
    def __init__(self, args):
        super(CIFARencoder, self).__init__()
        self._name = 'cifarE'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        convblock = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
        self.main = convblock
        self.linear = nn.Linear(4*4*4*self.dim, 4*4*5)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output
