import numpy as np
from torch import nn
from torch.nn import functional as F

class ThirdG(nn.Module):
    def __init__(self, args):
        super(ThirdG, self).__init__()
        self.shape = (128, 128, 3)
        self.dim = args.dim
        preprocess = nn.Sequential(
                nn.Linear(16*self.dim, 16 * 16 * 4 * self.dim),
                nn.BatchNorm2d(16 * 16 * 4 * self.dim),
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
        output = output.view(-1, 4 * self.dim, 16, 16)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 128, 128)


class ThirdE(nn.Module):
    def __init__(self, args):
        super(ThirdE, self).__init__()
        self._name = 'cifarE'
        self.shape = (128, 128, 3)
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
        self.linear = nn.Linear(16*16*4*self.dim, 4*4*self.dim)

    def forward(self, input, isUpsample = False):
        if isUpsample:
            input = self.upSample(input)
        output = self.main(input)
        output = output.view(-1, 16*16*4*self.dim)
        output = self.linear(output)
        return output

class ThirdD(nn.Module):
    def __init__(self, args):
        super(ThirdD, self).__init__()
        self._name = 'cifarD'
        self.shape = (128, 128, 3)
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
        self.linear = nn.Linear(16*16*4*self.dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 16*16*4*self.dim)
        output = self.linear(output)
        return output