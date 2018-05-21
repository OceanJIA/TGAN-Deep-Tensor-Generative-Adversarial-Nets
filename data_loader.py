import os
import numpy as np
import matplotlib.image as mpimg
import torch
import cv2
import torchvision.transforms as torchTrans
import utils
import torchvision


def stack_data(batch_size, _data, data_size):
    preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(data_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    datashape = (3, 32, 32)
    _data = _data.reshape(batch_size, *datashape).transpose(0, 2, 3, 1)
    real_data = torch.stack([preprocess(item) for item in _data]).cuda()

    return real_data

class BSDDataLoader():
    def __init__(self, dataType='BSD', batch_size=10, args=None):
        self.dataType = dataType
        if dataType == 'BSD':
            self.dataPath = './dataset/'
            self.imgList = os.listdir(self.dataPath)
            self.batchSize = batch_size
            self.len = len(self.imgList)
            self.loimgs = torch.zeros((300, 3, 32, 32))
            self.midImgs = torch.zeros((300, 3, 64, 64))
            self.HDImgs = torch.zeros((300, 3, 128, 128))
            self.iter = 0
            preprocess = torchTrans.Compose([
                torchTrans.ToTensor(),
                torchTrans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            for i in range(self.len):
                imgH = cv2.resize(mpimg.imread(self.dataPath + self.imgList[i + self.iter]), (128, 128))[:, :, 0:-1]
                imgM = cv2.resize(imgH, (64, 64))
                img = cv2.resize(imgM, (32, 32))
                imgH = preprocess(imgH)
                imgM = preprocess(imgM)
                img = preprocess(img)
                self.loimgs[i,:,:,:] = img
                self.midImgs[i,:,:,:] = imgM
                self.HDImgs[i,:,:,:] = imgH
        elif dataType == 'CIFAR':
            train_gen, dev_gen, test_gen = utils.dataset_iterator(args)
            self.batchSize = batch_size
            self.gen = utils.inf_train_gen(train_gen)
        elif dataType == 'PASCAL':
            self.dataPath = './VOCdevkit/VOC2012/'
            self.imgList = []
            for line in open(self.dataPath + 'ImageSets/Main/trainval.txt'):
                self.imgList.append(line[0:-1])

            self.batchSize = batch_size
            self.len = len(self.imgList)
            self.loimgs = torch.zeros((self.len, 3, 32, 32))
            self.midImgs = torch.zeros((self.len, 3, 64, 64))
            self.HDImgs = torch.zeros((self.len, 3, 128, 128))
            self.iter = 0
            preprocess = torchTrans.Compose([
                torchTrans.ToTensor(),
                torchTrans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            for i in range(self.len):
                imgH = cv2.resize(mpimg.imread(self.dataPath + 'JPEGImages/' + self.imgList[i + self.iter] + '.jpg'), (128, 128))
                imgM = cv2.resize(imgH, (64, 64))
                img = cv2.resize(imgH, (32, 32))
                imgH = preprocess(imgH)
                imgM = preprocess(imgM)
                img = preprocess(img)
                self.loimgs[i, :, :, :] = img
                self.midImgs[i, :, :, :] = imgM
                self.HDImgs[i, :, :, :] = imgH

    def length(self):
        if self.dataType == 'BSD' or self.dataType == 'PASCAL':
            return self.len / self.batchSize
        elif self.dataType == 'CIFAR':
            return 'Unknown'

    def getNextLoBatch(self):
        if self.dataType == 'BSD' or self.dataType == 'PASCAL':
            self.iter = self.iter + self.batchSize
            if self.iter + self.batchSize > self.len:
                self.iter = 0
            return self.loimgs[self.iter:self.iter+self.batchSize,:,:,:]
        elif self.dataType == 'CIFAR':
            return stack_data(self.batchSize, next(self.gen), (32, 32))

    def getNextMidBatch(self):
        if self.dataType == 'BSD' or self.dataType == 'PASCAL':
            self.iter = self.iter + self.batchSize
            if self.iter + self.batchSize > self.len:
                self.iter = 0
            return self.midImgs[self.iter:self.iter + self.batchSize, :, :, :]
        elif self.dataType == 'CIFAR':
            return stack_data(self.batchSize, next(self.gen), (64, 64))

    def getNextHDBatch(self):
        if self.dataType == 'BSD' or self.dataType == 'PASCAL':
            self.iter = self.iter + self.batchSize
            if self.iter + self.batchSize > self.len:
                self.iter = 0
            return self.HDImgs[self.iter:self.iter + self.batchSize, :, :, :]
        elif self.dataType == 'CIFAR':
            return stack_data(self.batchSize, next(self.gen), (128, 128))
