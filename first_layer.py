import numpy as np
from torch import nn
from torch.nn import functional as F

class FirstG(nn.Module):
    def __init__(self, args):
        super(FirstG, self).__init__()
        self._name = 'FirstG'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        preprocess = nn.Sequential(
                        nn.Linear(4*4*5, 4*4*5*self.dim),
                        nn.BatchNorm2d(4 * 4 * 5 * self.dim),
                        nn.ReLU(True),
                        ).cuda()
        block1 = nn.Sequential(
                nn.ConvTranspose2d(5 * self.dim, 2 * self.dim, 2, stride=2),
                nn.BatchNorm2d(2 * self.dim),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(2 * self.dim, self.dim, 2, stride=2),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
                )
        deconv_out = nn.ConvTranspose2d(self.dim, 3, 2, stride=2)

        self.preprocess = preprocess
        self.noiseProcess = nn.Sequential(
                        nn.Linear(4*4*5, 4*4*5),
                        nn.BatchNorm2d(4 * 4 * 5),
                        nn.ReLU(True),
                        ).cuda()
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input, isNoise=False):
        if isNoise:
            input = self.noiseProcess(input)
        output = self.preprocess(input)
        output = output.view(-1, 5 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)

class FirstE(nn.Module):
    def __init__(self, args):
        super(FirstE, self).__init__()
        self._name = 'FirstE'
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

class FirstD(nn.Module):
    def __init__(self, args):
        super(FirstD, self).__init__()
        self._name = 'FirstD'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        convblock = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.LeakyReLU(),
                )
        self.main = convblock
        self.linear = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output