import torch
from torchvision.models import resnet18
from torch.autograd import Variable
from torch import optim
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2

# 通过二阶导计算Hessian矩阵，但是，是在训练中反向传播loss时，而不是计算图像的
def second_order(model,input,target,optimizer):
    # as usual
    output = model(input)
    loss = torch.nn.functional.nll_loss(output, target)

    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    # torch.autograd.grad does not accumuate the gradients into the .grad attributes
    # It instead returns the gradients as Variable tuples.

    # now compute the 2-norm of the grad_params
    grad_norm = 0
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    grad_norm = grad_norm.sqrt()

    # take the gradients wrt grad_norm. backward() will accumulate
    # the gradients into the .grad attributes
    grad_norm.backward()

    # do an optimization step
    optimizer.step()

# 将图像处理为灰度图像
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# 图像的Hessian矩阵
def image_hessian(path):
    gray = cv2.imread(path, 0)
    # gray = rgb2gray(gray_)
    gray = gray/1.0
    print(gray.shape)
    H, W = gray.shape
    J = np.zeros((H, W))
    for i in range(2,H-1):
        for j in range(2, W-1):
            J[i,j] = 4*gray[i, j] - (gray[i+1, j] + gray[i-1, j] + gray[i, j+1] + gray[i, j-1])
    return J


# 图像的一阶导数
def one_order(path):
    I = cv2.imread(path, 0)
    H, W = I.shape
    M = I/1.0
    J = np.zeros((H, W))
    for i in range(2,H-1):
        for j in range(2, W-1):
            J[i,j] = abs(M[i-1, j+1] - M[i-1, j-1] + 2*M[i, j+1] - 2*M[i, j-1] + M[i+1, j+1]-M[i+1, j-1]) \
                     + abs(M[i-1, j-1] - M[i+1, j-1] + 2*M[i-1, j] - 2*M[i+1, j] + M[i-1, j+1]-M[i+1, j+1])
    return J


if __name__ == '__main__':
    path = "image/liver.png"
    # gray = mpimg.imread(path)
    gray = cv2.imread(path, 0)
    second = image_hessian(path)
    one = one_order(path)

    add2 = gray+second
    add1 = gray+one

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.imshow(gray)
    plt.title('raw image')
    plt.subplot(2, 2, 2)
    plt.imshow(one.astype(int)*10000)
    plt.title('second order image')
    plt.subplot(2, 2, 3)
    plt.imshow(one.astype(int)*10)
    plt.title('first order image')
    plt.subplot(2, 2, 4)
    plt.imshow(one.astype(int))
    plt.title('image')
    plt.show()

    # J = cv2.imread(path, 0)
    # gray_ = mpimg.imread(path)
    # plt.figure(1)
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title('raw image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(J)
    # plt.title('second order image')
    # plt.show()


    # # dummy inputs for the example
    # model = resnet18().cuda()
    # input = Variable(torch.randn(2, 3, 224, 224).cuda(), requires_grad=True)
    # target = Variable(torch.zeros(2).long().cuda())
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    #
    # second_order(model, input, target, optimizer)
