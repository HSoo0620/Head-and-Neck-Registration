import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import random
import numpy as np
import natsort

import nibabel as nib
import os 
import numpy as np
import torchio as tio
import torch.nn.functional as F
import torchvision.transforms as T

def resize_img(img, gt, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() 
    grid = grid.unsqueeze(0) 
    img = torch.Tensor(img).cpu()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = img[0][0]
    gt = torch.Tensor(gt).cpu()
    gt = gt.unsqueeze(0)
    gt = gt.unsqueeze(0)
    gt = gt.permute(0,1,4,3,2)
    gt = F.grid_sample(gt, grid, mode='nearest', align_corners=True)
    gt = gt[0][0]
    return img, gt

def HnN_resize_img(img, gt, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() 
    grid = grid.unsqueeze(0) 
    img = torch.Tensor(img).cpu()
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = img[0][0]
    gt = torch.Tensor(gt).cpu()
    # gt = gt.unsqueeze(0)
    gt = gt.unsqueeze(0) # 1 C D H W
    gt = gt.permute(0,1,4,3,2)
    gt = F.grid_sample(gt, grid, mode='nearest', align_corners=True)
    gt = gt[0]
    return img, gt


def volgen_datalist(vol_names, mask_names):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)
    
    if isinstance(mask_names, str):
        if os.path.isdir(mask_names):
            mask_names = os.path.join(mask_names, '*')
        mask_names = glob.glob(mask_names)

    return vol_names, mask_names 

def affine_augmentation(img, label):
    subject = tio.Subject(
            original=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=label))

    affine = tio.RandomAffine(scales=(0.75, 1.25), degrees=(5,5,5), translation=(20,5,5), label_interpolation = 'nearest')
    # composed_transform = tio.Compose([affine])
    transformed_subject = affine(subject)
    transformed_img = transformed_subject['original'].data
    transformed_label = transformed_subject['mask'].data
    return transformed_img, transformed_label


class HnNDataset_train(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        """
        input data shape : C(x[1], y[1], x_seg[class], y_seg[class]) D H W
        """
        path = self.paths[index]
        x = nib.load(path[0]).get_fdata()
        y = nib.load(path[1]).get_fdata()
        x_seg = nib.load(path[2]).get_fdata()
        y_seg = nib.load(path[3]).get_fdata()

        x, y = np.swapaxes(x, 0, 2), np.swapaxes(y, 0, 2)
        x_seg, y_seg = np.swapaxes(x_seg, 0, 3), np.swapaxes(y_seg, 0, 3)
        x_seg, y_seg = np.swapaxes(x_seg, 1, 2), np.swapaxes(y_seg, 1, 2)

        x, y = x[None, ...], y[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x_af, x_af_seg = HnN_resize_img(x, x_seg, 64, 80, 160)
        y_af, y_af_seg = HnN_resize_img(y, y_seg, 64, 80, 160)

        x_af = np.ascontiguousarray(x_af)  # [Bsize,channelsHeight,,Width,Depth]
        y_af = np.ascontiguousarray(y_af)
        x_af_seg = np.ascontiguousarray(x_af_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_af_seg = np.ascontiguousarray(y_af_seg)
        x_af, y_af, x_af_seg, y_af_seg = torch.from_numpy(x_af), torch.from_numpy(y_af), torch.from_numpy(x_af_seg), torch.from_numpy(y_af_seg)

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x_af, y_af = x_af[None, ...], y_af[None, ...]

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x_af, y_af, x_af_seg, y_af_seg, x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class HnNDataset_valid(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        """
        input data shape : C(x[1], y[1], x_seg[45], y_seg[45]) D H W
        """
        path = self.paths[index]
        x = nib.load(path[0]).get_fdata()
        y = nib.load(path[1]).get_fdata()
        x_seg = nib.load(path[2]).get_fdata()
        y_seg = nib.load(path[3]).get_fdata()

        x, y = np.swapaxes(x, 0, 2), np.swapaxes(y, 0, 2)
        x_seg, y_seg = np.swapaxes(x_seg, 0, 3), np.swapaxes(y_seg, 0, 3)
        x_seg, y_seg = np.swapaxes(x_seg, 1, 2), np.swapaxes(y_seg, 1, 2)

        x, y = x[None, ...], y[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x_af, x_af_seg = HnN_resize_img(x, x_seg, 64, 80, 160)
        y_af, y_af_seg = HnN_resize_img(y, y_seg, 64, 80, 160)

        x_af = np.ascontiguousarray(x_af)  # [Bsize,channelsHeight,,Width,Depth]
        y_af = np.ascontiguousarray(y_af)
        x_af_seg = np.ascontiguousarray(x_af_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_af_seg = np.ascontiguousarray(y_af_seg)
        x_af, y_af, x_af_seg, y_af_seg = torch.from_numpy(x_af), torch.from_numpy(y_af), torch.from_numpy(x_af_seg), torch.from_numpy(y_af_seg)

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x_af, y_af = x_af[None, ...], y_af[None, ...]
        
        x, y, x_seg, y_seg= torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x_af, y_af, x_af_seg, y_af_seg, x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class HnNDataset_affine(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        """
        input data shape : C(x[1], y[1], x_seg[45], y_seg[45]) D H W
        """
        path = self.paths[index]
        x = nib.load(path[0]).get_fdata()
        y = nib.load(path[1]).get_fdata()
        x_seg = nib.load(path[2]).get_fdata()
        y_seg = nib.load(path[3]).get_fdata()

        x, y = np.swapaxes(x, 0, 2), np.swapaxes(y, 0, 2)
        x_seg, y_seg = np.swapaxes(x_seg, 0, 3), np.swapaxes(y_seg, 0, 3)
        x_seg, y_seg = np.swapaxes(x_seg, 1, 2), np.swapaxes(y_seg, 1, 2)

        x, y = x[None, ...], y[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x, x_seg = HnN_resize_img(x, x_seg, 64, 80, 160)
        y, y_seg = HnN_resize_img(y, y_seg, 64, 80, 160)

        x = np.ascontiguousarray(x[None]) 
        y = np.ascontiguousarray(y[None])
        x_seg = np.ascontiguousarray(x_seg) 
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg= torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        x, x_seg = affine_augmentation(x, x_seg)
        y, y_seg = affine_augmentation(y, y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class HnNDataset_train(Dataset):
    def __init__(self, data_path, transforms, patch_size):
        self.paths = data_path
        self.transforms = transforms
        self.patch_size = patch_size

    def __getitem__(self, index):
        """
        input data shape : C(x[1], y[1], x_seg[class], y_seg[class]) D H W
        x_af, y_af, x_af_seg, y_af_seg: For Affine Registration
        x, y, x_seg, y_seg: For DIR Registration
        """
        d, h, w = self.patch_size
        path = self.paths[index]
        x = nib.load(path[0]).get_fdata()
        y = nib.load(path[1]).get_fdata()
        x_seg = nib.load(path[2]).get_fdata()
        y_seg = nib.load(path[3]).get_fdata()

        x, y = np.swapaxes(x, 0, 2), np.swapaxes(y, 0, 2)
        x_seg, y_seg = np.swapaxes(x_seg, 0, 3), np.swapaxes(y_seg, 0, 3)
        x_seg, y_seg = np.swapaxes(x_seg, 1, 2), np.swapaxes(y_seg, 1, 2)

        x, y = x[None, ...], y[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x_af, x_af_seg = HnN_resize_img(x, x_seg, d, h, w)
        y_af, y_af_seg = HnN_resize_img(y, y_seg, d, h, w)

        x_af = np.ascontiguousarray(x_af)  
        y_af = np.ascontiguousarray(y_af)
        x_af_seg = np.ascontiguousarray(x_af_seg)  
        y_af_seg = np.ascontiguousarray(y_af_seg)
        x_af, y_af, x_af_seg, y_af_seg = torch.from_numpy(x_af), torch.from_numpy(y_af), torch.from_numpy(x_af_seg), torch.from_numpy(y_af_seg)

        x = np.ascontiguousarray(x)  
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  
        y_seg = np.ascontiguousarray(y_seg)

        x_af, y_af = x_af[None, ...], y_af[None, ...]

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x_af, y_af, x_af_seg, y_af_seg, x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class HnNDataset_valid(Dataset):
    def __init__(self, data_path, transforms, patch_size):
        self.paths = data_path
        self.transforms = transforms
        self.patch_size = patch_size

    def __getitem__(self, index):
        """
        input data shape : C(x[1], y[1], x_seg[45], y_seg[45]) D H W
        x_af, y_af, x_af_seg, y_af_seg: For Affine Registration
        x, y, x_seg, y_seg: For DIR Registration
        """
        d, h, w = self.patch_size

        path = self.paths[index]
        x = nib.load(path[0]).get_fdata()
        y = nib.load(path[1]).get_fdata()
        x_seg = nib.load(path[2]).get_fdata()
        y_seg = nib.load(path[3]).get_fdata()

        x, y = np.swapaxes(x, 0, 2), np.swapaxes(y, 0, 2)
        x_seg, y_seg = np.swapaxes(x_seg, 0, 3), np.swapaxes(y_seg, 0, 3)
        x_seg, y_seg = np.swapaxes(x_seg, 1, 2), np.swapaxes(y_seg, 1, 2)

        x, y = x[None, ...], y[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x_af, x_af_seg = HnN_resize_img(x, x_seg, d, h, w)
        y_af, y_af_seg = HnN_resize_img(y, y_seg, d, h, w)

        x_af = np.ascontiguousarray(x_af)  # [Bsize,channelsHeight,,Width,Depth]
        y_af = np.ascontiguousarray(y_af)
        x_af_seg = np.ascontiguousarray(x_af_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_af_seg = np.ascontiguousarray(y_af_seg)
        x_af, y_af, x_af_seg, y_af_seg = torch.from_numpy(x_af), torch.from_numpy(y_af), torch.from_numpy(x_af_seg), torch.from_numpy(y_af_seg)

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x_af, y_af = x_af[None, ...], y_af[None, ...]
        
        x, y, x_seg, y_seg= torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x_af, y_af, x_af_seg, y_af_seg, x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
