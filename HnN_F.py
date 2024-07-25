import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from natsort import natsorted
from os.path import join
from os import listdir
from data import datasets, trans
from scipy.stats import multivariate_normal
import os, glob
import matplotlib.pyplot as plt

def make_Data_case(img_name, label_name):
    train_img_list = natsorted(
        [join(img_name, file_name) for file_name in listdir(img_name)]
    )
    train_label_list = natsorted(
        [join(label_name, file_name) for file_name in listdir(label_name)]
    )
    train_list = datasets.volgen_datalist(train_img_list, train_label_list)

    pairs = []
    for source in range(0, len(train_list[0])): #data_num
        for target in range(0, len(train_list[0])):
            if source == target:
                continue
            pairs.append((train_list[0][source], train_list[0][target], train_list[1][source], train_list[1][target]))
    return pairs


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, 72:88, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[1]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, i, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=3):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'), reverse=True)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[-1])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def crop_img_HnN(x, y, x_seg, y_seg, patch_size, img_size):
    patch_z_size, patch_y_size, patch_x_size = patch_size
    img_z_size, img_y_size, img_x_size = img_size 
    x, y, x_seg, y_seg = x.to("cuda"), y.to("cuda"), x_seg.to("cuda"), y_seg.to("cuda")

    start_point_x = torch.randint(0, img_x_size-patch_x_size-1, (1,))
    start_point_y = torch.randint(0, img_y_size-patch_y_size-1, (1,))
    start_point_z = torch.randint(0, img_z_size-patch_z_size-1, (1,))

    x = x[:,:,start_point_z:start_point_z+patch_z_size,start_point_y:start_point_y+patch_y_size,start_point_x:start_point_x+patch_x_size]
    y = y[:,:,start_point_z:start_point_z+patch_z_size,start_point_y:start_point_y+patch_y_size,start_point_x:start_point_x+patch_x_size]
    x_seg = x_seg[:,:,start_point_z:start_point_z+patch_z_size,start_point_y:start_point_y+patch_y_size,start_point_x:start_point_x+patch_x_size]
    y_seg = y_seg[:,:,start_point_z:start_point_z+patch_z_size,start_point_y:start_point_y+patch_y_size,start_point_x:start_point_x+patch_x_size]
    x, y, x_seg, y_seg = x.to("cuda"), y.to("cuda"), x_seg.to("cuda"), y_seg.to("cuda")
    
    return x, y, x_seg, y_seg

def save_img(img, img_name, mode='label') :

    if mode == 'label':
        img = torch.argmax(img, dim=1).float()   
    img = img.squeeze() # d h w 
    img = img.cpu().detach().numpy()
    img = np.swapaxes(img,0,2) #  w h d
    x = nib.Nifti1Image(img, None)
    nib.save(x, img_name)

def resize_img(img, gt, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    h = torch.stack((meshz, meshy, meshx), 3).cpu() 
    grid = grid.unsqueeze(0) 
    img = img.cpu()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    gt = gt.cpu()
    gt = gt.unsqueeze(0)
    gt = gt.unsqueeze(0)
    gt = gt.permute(0,1,4,3,2)
    gt = F.grid_sample(gt, grid, mode='nearest', align_corners=True)
    return img, gt

def resize_dvf(img, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() 
    grid = grid.unsqueeze(0) 
    img = img.cpu()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = img[0][0]
    return img

def restore_dvf(flow, d, h, w):
    upsampling_flow_z = resize_dvf(flow[0][0], d, h, w)
    upsampling_flow_y = resize_dvf(flow[0][1], d, h, w)
    upsampling_flow_x = resize_dvf(flow[0][2], d, h, w)
    
    upsampling_flow_z *= (d / 64)
    upsampling_flow_y *= (h / 80)
    upsampling_flow_x *= (w / 160)
            
    upsampling_flow = torch.cat([upsampling_flow_z[None], upsampling_flow_y[None], upsampling_flow_x[None]], dim=0) # (c, w, h, d)
    upsampling_flow = upsampling_flow[None] # (1, c, w, h, d)
    return upsampling_flow


def make_gaussian_kernel(patch_size):
    patch_z_size, patch_y_size, patch_x_size = patch_size

    z, y, x = np.mgrid[-1.0:1.0:patch_z_size, -1.0:1.0:patch_y_size, -1.0:1.0:patch_x_size]
    
    zyx = np.column_stack([z.flat, y.flat, x.flat])
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([.25, .25, .25])
    covariance = np.diag(sigma**2)
    gaussian_kernel = multivariate_normal.pdf(zyx, mean=mu, cov=covariance)
    gaussian_kernel = gaussian_kernel.reshape(x.shape)
    gaussian_kernel = (gaussian_kernel - gaussian_kernel.min()) / (gaussian_kernel.max() - gaussian_kernel.min()) + 0.0001 # 1.0001 ~ 0.0001
    gaussian_kernel = torch.Tensor(gaussian_kernel)
    gaussian_kernel = torch.stack([gaussian_kernel, gaussian_kernel, gaussian_kernel], dim=-1)
    return gaussian_kernel