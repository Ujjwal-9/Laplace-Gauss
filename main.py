

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2



img = cv2.imread('test.jpg', 0)
img.shape



def plot(img):
    fig = plt.imshow(img, cmap = 'gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)



img = torch.Tensor(img)
img = (img.unsqueeze(0)).unsqueeze(0)
img.shape




kernel = torch.Tensor([[[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]]])
kernel = kernel/torch.sum(kernel)
kernel.shape




src = autograd.Variable(img)
kernel = autograd.Variable(kernel)
pyramid_size = 7



print("src shape: ", src.size())
print("kernel shape : ", kernel.size())



def pad(input, padding, kind, k_h, k_w, s_h=1, s_w=1, dilation=1):
    if padding == 'VALID':
        return input
    elif padding == 'SAME' and kind in ('conv2d', 'pool2d'):
        in_height, in_width = input.size(2), input.size(3)
        out_height = int(np.ceil(float(in_height) / float(s_h)))
        out_width  = int(np.ceil(float(in_width) / float(s_w)))

        pad_along_height = max((out_height - 1) * s_h + k_h - in_height, 0)
        pad_along_width = max((out_width - 1) * s_w + k_w - in_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom))
        return input



out = []
out.append(src)
pool = nn.AvgPool2d(kernel_size=(3,3), stride=(2,2))
for i in range(pyramid_size):
    ops = F.conv2d(pad(src, 'SAME', 'conv2d', 5, 5), kernel)
    src = pool(ops)
    out.append(src)



def for_image(i, out):
    test =  out[i].view(-1, out[i].size()[-1])
    test = test.data.numpy()
    return test



def image(img):
    img = img.view(-1, img.size(-1))
    return img.data.numpy()



plot(for_image(3, out))



l_out = []
for i in range(1,pyramid_size-1):
    ops = F.conv2d(out[i+1], kernel)
    ops = F.upsample(ops, tuple(out[i].size()[2:]), mode='bilinear')
    ops = (ops.unsqueeze(0)).unsqueeze(0)
    L = out[i] - ops
    l_out.append(L)



plot(for_image(3, l_out))