import os
import sys
import time
import argparse
import numpy as np
from scipy.misc import imshow

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F

import ops
import plot
import utils
import encoders
import generators
import discriminators
from data import mnist
from data import cifar10
from dataLoader import BSDDataLoader
from tensorGen import generateTensor
from inception_score import inception_score

import SecondLayer


def load_args():
    parser = argparse.ArgumentParser(description='aae-wgan')
    parser.add_argument('-d', '--dim', default=32, type=int, help='latent space size')
    parser.add_argument('-l', '--gp', default=64, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=10, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-o', '--output_dim', default=3072, type=int)
    parser.add_argument('--dataset', default='cifar10')
    args = parser.parse_args()
    return args


def load_models(args):
    netG = generators.CIFARgenerator(args).cuda()
    netD = discriminators.CIFARdiscriminator(args).cuda()
    netE = encoders.CIFARencoder(args).cuda()

    print (netG, netD, netE)
    return (netG, netD, netE)


def stack_data(args, _data):
    if args.dataset == 'cifar10':
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        datashape = (3, 32, 32)
        _data = _data.reshape(args.batch_size, *datashape).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data]).cuda()
    elif args.dataset == 'mnist':
        real_data = torch.Tensor(_data).cuda()

    return real_data


def train():
    args = load_args()
    train_gen, dev_gen, test_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    netG, _, netE = load_models(args)
    SecondG = SecondLayer.SecondG(args).cuda()
    SecondD = SecondLayer.SecondD(args).cuda()
    SecondE = SecondLayer.SecondE(args).cuda()
    incep_score = 0
    netG.load_state_dict(torch.load('./1stLayer/1stLayerG71999.model'))
    # SecondG.load_state_dict(torch.load('./2ndLayer/2ndLayerG18999.model'))
    # SecondE.load_state_dict(torch.load('./2ndLayer/2ndLayerE18999.model'))
    # SecondD.load_state_dict(torch.load('./2ndLayer/2ndLayerD18999.model'))

    optimizerD = optim.Adam(SecondD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(SecondG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerE = optim.Adam(SecondE.parameters(), lr=1e-4, betas=(0.5, 0.9))
    ae_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()

    dataLoader = BSDDataLoader('CIFAR', args.batch_size, args)

    for iteration in range(args.epochs):
        start_time = time.time()
        """ Update AutoEncoder """
        for p in SecondD.parameters():
            p.requires_grad = False
        SecondG.zero_grad()
        SecondE.zero_grad()
        tic = time.time()
        real_data = dataLoader.getNextMidBatch().cuda()
        toc = time.time()
        # print(toc - tic)
        real_data_v = autograd.Variable(real_data)
        encoding = SecondE(real_data_v)
        fake = SecondG(encoding)
        ae_loss = ae_criterion(fake, real_data_v)
        ae_loss.backward(one)
        optimizerE.step()
        optimizerG.step()

        """ Update D network """

        for p in SecondD.parameters():
            p.requires_grad = True
        for i in range(5):
            # _data = next(gen)
            real_data = dataLoader.getNextMidBatch().cuda()
            # real_data = stack_data(args, _data)
            real_data_v = autograd.Variable(real_data)
            # train with real data
            SecondD.zero_grad()
            D_real = SecondD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
            # train with fake data
            noise = generateTensor(args.batch_size).cuda()
            noisev = autograd.Variable(noise, volatile=True)
            fake = autograd.Variable(SecondG(SecondE(netG(noisev, True), True)).data)
            inputv = fake
            D_fake = SecondD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = ops.calc_gradient_penalty(args,
                                                         SecondD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        # Update generator network (GAN)
        noise = generateTensor(args.batch_size).cuda()
        noisev = autograd.Variable(noise)
        fake = SecondG(SecondE(netG(noisev, True), True))
        # fake = SecondG(noisev)
        G = SecondD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        # Write logs and save samples
        save_dir = './plots/' + args.dataset
        plot.plot(save_dir, '/disc cost', D_cost.cpu().data.numpy())
        plot.plot(save_dir, '/gen cost', G_cost.cpu().data.numpy())
        plot.plot(save_dir, '/w1 distance', Wasserstein_D.cpu().data.numpy())
        plot.plot(save_dir, '/ae cost', ae_loss.data.cpu().numpy())

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 1000 == 999:
            dev_disc_costs = []
            for images, _ in dev_gen():
                imgs = dataLoader.getNextMidBatch().cuda()
                imgs_v = autograd.Variable(imgs, volatile=True)
                D = SecondD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            plot.plot(save_dir, '/dev disc cost', np.mean(dev_disc_costs))
            torch.save(netG.state_dict(), './1stLayer/1stLayerG%d_2.model' % iteration)
            torch.save(SecondE.state_dict(), './2ndLayer/2ndLayerE%d.model' % iteration)
            torch.save(SecondG.state_dict(), './2ndLayer/2ndLayerG%d.model' % iteration)
            torch.save(SecondD.state_dict(), './2ndLayer/2ndLayerD%d.model' % iteration)
            # utils.generate_image(iteration, netG, save_dir, args)
            utils.generate_MidImage(iteration, netG, SecondE, SecondG, save_dir, args)
            # utils.generate_MidAE_image(iteration, SecondE, SecondG, save_dir, args, real_data_v)



        if iteration % 2000 == 1999:
            noise = generateTensor(args.batch_size).cuda()
            noisev = autograd.Variable(noise, volatile=True)
            fake = autograd.Variable(SecondG(SecondE(netG(noisev, True), True)).data)
            incep_score = (inception_score(fake.data.cpu().numpy(), resize=True, batch_size=5)[0])

        endtime = time.time()
        # Save logs every 100 iters
        # if (iteration < 5) or (iteration % 1000 == 999):
        #     plot.flush()
        #     plot.tick()
        print('iter:', iteration, 'total time %4f' % (endtime-start_time), 'ae loss %4f' % ae_loss.data[0],
                        'G cost %4f' % G_cost.data[0], 'inception score %4f' % incep_score)


if __name__ == '__main__':
    train()