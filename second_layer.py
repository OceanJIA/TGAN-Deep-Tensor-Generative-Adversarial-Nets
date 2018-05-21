import numpy as np
from torch import nn
from torch.nn import functional as F

class SecondG(nn.Module):
    def __init__(self, args):
        super(SecondG, self).__init__()
        self.shape = (64, 64, 3)
        self.dim = args.dim
        preprocess = nn.Sequential(
                nn.Linear(2*2*self.dim, 8 * 8 * 4 * self.dim),
                nn.BatchNorm2d(8 * 8 * 4 * self.dim),
                nn.ReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 2, stride=2),
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
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 8, 8)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 64, 64)


class SecondE(nn.Module):
    def __init__(self, args):
        super(SecondE, self).__init__()
        self._name = 'cifarE'
        self.shape = (64, 64, 3)
        self.dim = args.dim
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
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
        self.linear = nn.Linear(8*8*4*self.dim, 2*2*self.dim)

    def forward(self, input, isUpsample = False):
        if isUpsample:
            input = self.upSample(input)
        output = self.main(input)
        output = output.view(-1, 8*8*4*self.dim)
        output = self.linear(output)
        return output

class SecondD(nn.Module):
    def __init__(self, args):
        super(SecondD, self).__init__()
        self._name = 'cifarD'
        self.shape = (64, 64, 3)
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
        self.linear = nn.Linear(8*8*4*self.dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 8*8*4*self.dim)
        output = self.linear(output)
        return output