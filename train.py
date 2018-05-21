import os
import sys
import time
import argparse
import numpy as np

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F

import ops
import utils
import first_layer
from data_loader import BSDDataLoader
from tensorGen import generateTensor
from inception_score import inception_score


def load_args():
    parser = argparse.ArgumentParser(description='TGAN')
    parser.add_argument('-d', '--dim', default=32, type=int, help='latent space size')
    parser.add_argument('-l', '--gp', default=64, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=10, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-o', '--output_dim', default=3072, type=int)
    parser.add_argument('--dataset', default='PASCAL')
    args = parser.parse_args()
    return args

def train():
    args = load_args()
    torch.manual_seed(1)
    netG = first_layer.FirstG(args).cuda()
    netD = first_layer.FirstD(args).cuda()
    netE = first_layer.FirstE(args).cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerE = optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
    ae_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()

    dataLoader = BSDDataLoader(args.dataset, args.batch_size, args)
    incep_score = 0
    zeros = autograd.Variable(torch.zeros(args.batch_size, 4 * 4 * 5).cuda())

    for iteration in range(args.epochs):
        start_time = time.time()
        """ Update AutoEncoder """
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        netE.zero_grad()
        real_data = dataLoader.getNextLoBatch().cuda()
        real_data_v = autograd.Variable(real_data)
        encoding = netE(real_data_v)
        fake = netG(encoding)
        ae_loss = ae_criterion(fake, real_data_v) + ae_criterion(encoding, zeros)
        ae_loss.backward(one)
        optimizerE.step()
        optimizerG.step()

        """ Update D network """

        for p in netD.parameters():
            p.requires_grad = True
        for i in range(5):
            real_data = dataLoader.getNextLoBatch().cuda()
            real_data_v = autograd.Variable(real_data)
            # train with real data
            netD.zero_grad()
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
            # train with fake data
            noise = generateTensor(args.batch_size).cuda()
            noisev = autograd.Variable(noise, volatile=True)
            fake = autograd.Variable(netG(noisev, True).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = ops.calc_gradient_penalty(args,
                                                         netD, real_data_v.data, fake.data)
            gradient_penalty.backward()
            optimizerD.step()

        # Update generator network (GAN)
        noise = generateTensor(args.batch_size).cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev, True)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        # Write logs and save samples
        save_dir = './plots/' + args.dataset

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 1000 == 999:
            torch.save(netE.state_dict(), './1stLayer/1stLayerE%d.model' % iteration)
            torch.save(netG.state_dict(), './1stLayer/1stLayerG%d.model' % iteration)
            utils.generate_image(iteration, netG, save_dir, args)
        endtime = time.time()

        if iteration % 2000 == 1999:
            noise = generateTensor(1000).cuda()
            noisev = autograd.Variable(noise, volatile=True)
            fake = autograd.Variable(netG(noisev, True).data)
            incep_score = (inception_score(fake.data.cpu().numpy(), resize=True, batch_size=5))[0]

        print('iter:', iteration, 'total time %4f' % (endtime - start_time), 'ae loss %4f' % ae_loss.data[0],
                  'G cost %4f' % G_cost.data[0], 'inception score %4f' % incep_score)


if __name__ == '__main__':
    train()
