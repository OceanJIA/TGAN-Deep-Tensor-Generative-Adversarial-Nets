import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

class CIFARgenerator(nn.Module):
    def __init__(self, args):
        super(CIFARgenerator, self).__init__()
        self._name = 'cifarG'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        # preprocess = []
        # for i in range(5):
        #     preprocess.append(nn.Sequential(
        #             nn.Linear(4*4, 4*4*self.dim),
        #             nn.BatchNorm2d(4*4*self.dim),
        #             nn.ReLU(True),
        #             ).cuda())
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
        # outputList = []
        # for i in range(5):
        #     # print(input[:,4*4*i:4*4*(i+1)].shape)
        #     outputList.append(self.preprocess[i](input[:,4*4*i:4*4*(i+1)]))
        # output = torch.cat(input)
        if isNoise:
            input = self.noiseProcess(input)
        output = self.preprocess(input)
        output = output.view(-1, 5 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)
