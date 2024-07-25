import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn


class wgan_exp_loss(torch.nn.Module):

    def __init__(self):
        super(wgan_exp_loss, self).__init__()

    def forward(self, loss):
        min_loss = torch.tensor([0.0]).cuda()
        loss = max(min_loss, 1-torch.exp(loss))

        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


        
class Dice(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self, bg):
        super(Dice, self).__init__()
        self.bg = bg

    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2)) 

        y_true = y_true[:, self.bg:, :, :, :]
        y_pred = y_pred[:, self.bg:, :, :, :]

        top = 2 * ((y_true * y_pred).sum())
        bottom = torch.clamp(((y_true + y_pred)).sum(), min=1e-5)
        dice = ((1-(top / bottom)))
        return dice

class Dice_avg(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self, bg):
        super(Dice_avg, self).__init__()
        self.bg = bg

    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2)) 

        y_true = y_true[:, self.bg:, :, :, :]
        y_pred = y_pred[:, self.bg:, :, :, :]

        top = 2 * ((y_true * y_pred).sum(dim=vol_axes))
        bottom = torch.clamp(((y_true + y_pred)).sum(dim=vol_axes), min=1e-5)
        dice = ((1-(top / bottom)))
        return torch.mean(dice), dice


class MSE(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        with torch.no_grad():
            mask_t = torch.where(y_true<0.0, torch.tensor(0, dtype=y_true.dtype).cuda(), torch.tensor(1, dtype=y_true.dtype).cuda())
            mask_p = torch.where(y_pred<0.0, torch.tensor(0, dtype=y_pred.dtype).cuda(), torch.tensor(1, dtype=y_pred.dtype).cuda())
            mask = torch.logical_and(mask_t, mask_p)

        diff = y_true - y_pred

        return torch.mean((torch.square(diff)*mask).sum(dim=[1,2,3,4]) / ((mask).sum(dim=[1,2,3,4]) + 1e-5))

class Determ(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(Determ, self).__init__()

    def forward(self, y_pred): # y_pred: DVF. 1, 3, 64, 128, 128
        size = y_pred.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()

        y_pred = grid + y_pred

        y_pred = y_pred.permute(0, 2, 3, 4, 1) # 1, 64, 128, 128, 3

        J = np.gradient(y_pred.detach().cpu().numpy(), axis=(1, 2, 3))
        dx = J[0]
        dy = J[1]
        dz = J[2]

        dx = torch.Tensor(dx).cuda()
        dy = torch.Tensor(dy).cuda()
        dz = torch.Tensor(dz).cuda()
        
        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        dets_original =  Jdet0 - Jdet1 + Jdet2

        dets_neg = torch.clamp(dets_original, max=0.0)

        return torch.mean(dets_neg*dets_neg)


class NCC_HnN(torch.nn.Module):
    def __init__(self, win=None):
        super(NCC_HnN, self).__init__()
        self.win = win

    def forward(self, out, y_pred):

        I = out
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        # compute filters
        sum_filt = torch.ones([1, 1, *self.win]).to("cuda")

        pad_no = 0

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(-cc) 
