# coding=utf-8

import torch
import numpy as np
import time


def generateTensor(batchSize):
    A = torch.randn(batchSize, 4, 3, 5)
    B = torch.randn(batchSize, 3, 4, 5)
    C = torch.randn(batchSize, 5, 4, 4)
    # A1 = torch.zeros(batchSize, 4, 3, 5)
    # A2 = torch.zeros(batchSize, 4, 3, 5)
    # A3 = torch.zeros(batchSize, 4, 3, 5)
    # A4 = torch.zeros(batchSize, 4, 3, 5)
    # A1[:, :, :, 0] = A[:, :, :, 4]
    # A1[:, :, :, 1:5] = A[:, :, :, 0:4]
    # A2[:, :, :, 0] = A1[:, :, :, 4]
    # A2[:, :, :, 1:5] = A1[:, :, :, 0:4]
    # A3[:, :, :, 0] = A2[:, :, :, 4]
    # A3[:, :, :, 1:5] = A2[:, :, :, 0:4]
    # A4[:, :, :, 0] = A3[:, :, :, 4]
    # A4[:, :, :, 1:5] = A3[:, :, :, 0:4]
    #
    # C = torch.zeros(batchSize, 4, 4, 5)
    # for i in range(batchSize):
    #     C0 = torch.matmul(A[i,:,:, 0], B[i,:,:, 0]) + torch.matmul(A[i,:,:, 1], B[i,:,:, 1]) + torch.matmul(A[i,:,:, 2],
    #                 B[i,:,:, 2]) + torch.matmul(A[i,:,:, 3], B[i,:,:, 3]) + torch.matmul(A[i,:,:, 4], B[i,:,:, 4])
    #     C1 = torch.matmul(A1[i,:,:, 0], B[i,:,:, 0]) + torch.matmul(A1[i,:,:, 1], B[i,:,:, 1]) + torch.matmul(A1[i,:,:, 2],
    #                 B[i,:,:, 2]) + torch.matmul(A1[i,:,:, 3], B[i,:,:, 3]) + torch.matmul(A1[i,:,:, 4], B[i,:,:, 4])
    #     C2 = torch.matmul(A2[i,:,:, 0], B[i,:,:, 0]) + torch.matmul(A2[i,:,:, 1], B[i,:,:, 1]) + torch.matmul(A2[i,:,:, 2],
    #                 B[i,:,:, 2]) + torch.matmul(A2[i,:,:, 3], B[i,:,:, 3]) + torch.matmul(A2[i,:,:, 4], B[i,:,:, 4])
    #     C3 = torch.matmul(A3[i,:,:, 0], B[i,:,:, 0]) + torch.matmul(A3[i,:,:, 1], B[i,:,:, 1]) + torch.matmul(A3[i,:,:, 2],
    #                 B[i,:,:, 2]) + torch.matmul(A3[i,:,:, 3], B[i,:,:, 3]) + torch.matmul(A3[i,:,:, 4], B[i,:,:, 4])
    #     C4 = torch.matmul(A4[i,:,:, 0], B[i,:,:, 0]) + torch.matmul(A4[i,:,:, 1], B[i,:,:, 1]) + torch.matmul(A4[i,:,:, 2],
    #                 B[i,:,:, 2]) + torch.matmul(A4[i,:,:, 3], B[i,:,:, 3]) + torch.matmul(A4[i,:,:, 4], B[i,:,:, 4])
    #
    #     C[i, :, :, 0] = C0
    #     C[i, :, :, 1] = C1
    #     C[i, :, :, 2] = C2
    #     C[i, :, :, 3] = C3
    #     C[i, :, :, 4] = C4

    return C.view(-1, batchSize, 4*4*5)[0]


# if __name__ == '__main__':
#     tic = time.time()
#     print generateTensor(10).shape
#     toc = time.time()
#     print toc - tic