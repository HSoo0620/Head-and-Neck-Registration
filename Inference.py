from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted

from models.M_Adv import CONFIGS as CONFIGS_TM
import models.M_Adv as M_Adv
import nibabel as nib
from einops.einops import rearrange
from os.path import join
from os import listdir
import natsort
from scipy.stats import multivariate_normal

from models.TransMorph_Origin_Affine import CONFIGS as AFF_CONFIGS_TM
import models.TransMorph_Origin_Affine as TransMorph_affine
import torch.nn.functional as F

from HnN_F import make_Data_case, make_gaussian_kernel, save_img, restore_dvf
import argparse

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")    

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False

def main(args):
    
    '''
    Load Affine Model
    '''
    
    affine_model_dir = args.affine_model
    config_affine = AFF_CONFIGS_TM['TransMorph-Affine']
    affine_model = TransMorph_affine.SwinAffine(config_affine)
    affine_model.load_state_dict(torch.load('experiments/'+ affine_model_dir + natsorted(os.listdir('experiments/'+affine_model_dir), reverse=True)[0])['state_dict'])
    print('Affine Model: {} loaded!'.format(natsorted(os.listdir('experiments/'+ affine_model_dir), reverse=True)[0]))
    affine_model.cuda()
    affine_model.eval()
    for param in affine_model.parameters():
        param.requires_grad_(False)

    AffInfer_near = TransMorph_affine.ApplyAffine(mode='nearest')
    AffInfer_near.cuda()
    AffInfer_bi = TransMorph_affine.ApplyAffine(mode='bilinear')
    AffInfer_bi.cuda()

    save_dir = args.dir_model
    save_result_dir = 'results/' + save_dir
    if not os.path.exists(save_result_dir):
           os.makedirs(save_result_dir)

    '''
    Initialize model
    '''
    config = CONFIGS_TM['M-Adv']
    model = M_Adv.M_Adv_model(config)
    model.cuda()
    
    model_dir = args.dir_model
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
    print('M-Adv Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
    model.load_state_dict(best_model)
    
    '''
    Initialize spatial transformation function
    '''
    patch_size = args.patch_size 
    img_size = args.img_size
    save_validation_img = args.save_validation_img
    img_z_size, img_y_size, img_x_size = img_size 
    patch_z_size, patch_y_size, patch_x_size = patch_size
    sliding_stride = args.sliding_stride
    
    reg_model_bilin_origin = utils.register_model((img_z_size, img_y_size, img_x_size), 'bilinear')
    reg_model_bilin_origin.cuda()
    reg_model_bilin_patch = utils.register_model((patch_z_size, patch_y_size, patch_x_size), 'bilinear')
    reg_model_bilin_patch.cuda()

    reg_model_origin = utils.register_model((img_z_size, img_y_size, img_x_size), 'nearest')
    reg_model_origin.cuda()
    reg_model_patch = utils.register_model((patch_z_size, patch_y_size, patch_x_size), 'nearest')
    reg_model_patch.cuda()
    
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    img_name = args.dataset_dir + 'test/image/'
    label_name = args.dataset_dir + 'test/label/'
    test_pairs = make_Data_case(img_name, label_name)
    print("===Make Test Case : ", len(test_pairs), " Combinations")

    gaussian_kernel = make_gaussian_kernel(patch_size)
    
    val_set = datasets.HnNDataset_valid(test_pairs, transforms=test_composed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
   
    criterion_dsc = losses.Dice(bg=1)    
    criterion_avg_dsc = losses.Dice_avg(bg=1)

    # evaluate each class
    eval_avg_dsc = utils.AverageMeter()
    eval_dsc = utils.AverageMeter()
    eval_0 = utils.AverageMeter()
    eval_1 = utils.AverageMeter()
    eval_2 = utils.AverageMeter()
    eval_3 = utils.AverageMeter()
    eval_4 = utils.AverageMeter()
    eval_5 = utils.AverageMeter()
    eval_6 = utils.AverageMeter()
    eval_7 = utils.AverageMeter()
    eval_8 = utils.AverageMeter()
    eval_9 = utils.AverageMeter()
    eval_10 = utils.AverageMeter()
    eval_11 = utils.AverageMeter()
    eval_12 = utils.AverageMeter()
    eval_13 = utils.AverageMeter()
    eval_14 = utils.AverageMeter()
    eval_15 = utils.AverageMeter()
    eval_16 = utils.AverageMeter()
    eval_17 = utils.AverageMeter()
    eval_18 = utils.AverageMeter()
    eval_19 = utils.AverageMeter()
    eval_20 = utils.AverageMeter()
    eval_21 = utils.AverageMeter()
    eval_22 = utils.AverageMeter()

    with torch.no_grad():
        eval_idx = 0
        for data in val_loader:
            eval_idx +=1
            model.eval()
            data = [t.cuda() for t in data]
        
            x_af = data[0]
            y_af = data[1]
            
            x_af_in = torch.cat((x_af, y_af), dim=1)
            with torch.no_grad():
                _, affine_mat, _, _, _, _ = affine_model(x_af_in)

            x = AffInfer_bi(data[4].float(), affine_mat)
            y = data[5].float()            
            x_seg = AffInfer_near(data[6].float(), affine_mat)
            y_seg = data[7].float()

            total_DVF = torch.zeros(img_z_size, img_y_size, img_x_size,3).cuda()
            total_divid = torch.zeros(img_z_size, img_y_size, img_x_size,3).cuda()
            dvf_num=0

            for d in range(2):
                for h in range(2):
                    for w in range(3):
                        dvf_num += 1
                        sum_divid = torch.ones((patch_z_size, patch_y_size, patch_x_size, 3)).cuda()

                        x_buf_img = x[:,:,d*patch_z_size:d*patch_z_size+sliding_stride[0], h*patch_y_size:h*patch_y_size+sliding_stride[1], w*patch_x_size:w*patch_x_size+sliding_stride[2]]
                        y_buf_img = y[:,:,d*patch_z_size:d*patch_z_size+sliding_stride[0], h*patch_y_size:h*patch_y_size+sliding_stride[1], w*patch_x_size:w*patch_x_size+sliding_stride[2]]

                        x_in = torch.cat((x_buf_img.cuda(), y_buf_img.cuda()), dim=1)
                        _, flow, _, _, _, _ = model(x_in)

                        flow = restore_dvf(flow, patch_z_size, patch_y_size, patch_x_size)
                        buf_dvf = rearrange(flow[0], 'c d h w ->d h w c').cuda()
                        total_DVF[d*patch_z_size:d*patch_z_size+sliding_stride[0], h*patch_y_size:h*patch_y_size+sliding_stride[1], w*patch_x_size:w*patch_x_size+sliding_stride[2],:]+= buf_dvf * gaussian_kernel
                        total_divid[d*patch_z_size:d*patch_z_size+sliding_stride[0], h*patch_y_size:h*patch_y_size+sliding_stride[1], w*patch_x_size:w*patch_x_size+sliding_stride[2],:] += sum_divid * gaussian_kernel

                        del x_buf_img, y_buf_img, x_in, flow, buf_dvf, _
                        
            total_div_result = total_DVF / total_divid
            total_div_result = rearrange(total_div_result[None], 'b d h w c -> b c d h w')

            out_seg = reg_model_origin([x_seg.float(), total_div_result.float()])
            out = reg_model_bilin_origin([x.float(), total_div_result.float()])

            dsc = 1-criterion_dsc(out_seg, y_seg)
            eval_dsc.update(dsc.item(), x.size(0))

            avg_dsc, avg_list = criterion_avg_dsc(out_seg, y_seg)
            avg_list = avg_list[0]
            avg_dsc = 1-avg_dsc
            eval_avg_dsc.update(avg_dsc.item(), x.size(0))
            
            eval_0.update(1-avg_list[0].item(), x.size(0))
            eval_1.update(1-avg_list[1].item(), x.size(0))
            eval_2.update(1-avg_list[2].item(), x.size(0))
            eval_3.update(1-avg_list[3].item(), x.size(0))
            eval_4.update(1-avg_list[4].item(), x.size(0))
            eval_5.update(1-avg_list[5].item(), x.size(0))
            eval_6.update(1-avg_list[6].item(), x.size(0))
            eval_7.update(1-avg_list[7].item(), x.size(0))
            eval_8.update(1-avg_list[8].item(), x.size(0))
            eval_9.update(1-avg_list[9].item(), x.size(0))
            eval_10.update(1-avg_list[10].item(), x.size(0))
            eval_11.update(1-avg_list[11].item(), x.size(0)) 
            eval_12.update(1-avg_list[12].item(), x.size(0)) 
            eval_13.update(1-avg_list[13].item(), x.size(0)) 
            eval_14.update(1-avg_list[14].item(), x.size(0)) 
            eval_15.update(1-avg_list[15].item(), x.size(0)) 
            eval_16.update(1-avg_list[16].item(), x.size(0)) 
            eval_17.update(1-avg_list[17].item(), x.size(0)) 
            eval_18.update(1-avg_list[18].item(), x.size(0)) 
            eval_19.update(1-avg_list[19].item(), x.size(0)) 
            eval_20.update(1-avg_list[20].item(), x.size(0)) 
            eval_21.update(1-avg_list[21].item(), x.size(0)) 
            eval_22.update(1-avg_list[22].item(), x.size(0)) 


            print('Idx {} of Val {} DSC:{: .4f}'.format(eval_idx, len(val_loader),dsc.item()))
            if eval_idx == 30:
                print("--everage Dice eval_dsc: {:.5f} +- {:.3f}".format(eval_dsc.avg, eval_dsc.std))
                print("Dice eval_avg_dsc0: {:.5f} +- {:.3f}".format(eval_0.avg, eval_0.std))
                print("Dice eval_avg_dsc1: {:.5f} +- {:.3f}".format(eval_1.avg, eval_1.std))
                print("Dice eval_avg_dsc2: {:.5f} +- {:.3f}".format(eval_2.avg, eval_2.std))
                print("Dice eval_avg_dsc3: {:.5f} +- {:.3f}".format(eval_3.avg, eval_3.std))
                print("Dice eval_avg_dsc4: {:.5f} +- {:.3f}".format(eval_4.avg, eval_4.std))
                print("Dice eval_avg_dsc5: {:.5f} +- {:.3f}".format(eval_5.avg, eval_5.std))
                print("Dice eval_avg_dsc6: {:.5f} +- {:.3f}".format(eval_6.avg, eval_6.std))
                print("Dice eval_avg_dsc7: {:.5f} +- {:.3f}".format(eval_7.avg, eval_7.std))
                print("Dice eval_avg_dsc8: {:.5f} +- {:.3f}".format(eval_8.avg, eval_8.std))
                print("Dice eval_avg_dsc9: {:.5f} +- {:.3f}".format(eval_9.avg, eval_9.std))
                print("Dice eval_avg_dsc10: {:.5f} +- {:.3f}".format(eval_10.avg, eval_10.std))
                print("Dice eval_avg_dsc11: {:.5f} +- {:.3f}".format(eval_11.avg, eval_11.std))
                print("Dice eval_avg_dsc12: {:.5f} +- {:.3f}".format(eval_12.avg, eval_12.std))
                print("Dice eval_avg_dsc13: {:.5f} +- {:.3f}".format(eval_13.avg, eval_13.std))
                print("Dice eval_avg_dsc14: {:.5f} +- {:.3f}".format(eval_14.avg, eval_14.std))
                print("Dice eval_avg_dsc15: {:.5f} +- {:.3f}".format(eval_15.avg, eval_15.std))
                print("Dice eval_avg_dsc16: {:.5f} +- {:.3f}".format(eval_16.avg, eval_16.std))
                print("Dice eval_avg_dsc17: {:.5f} +- {:.3f}".format(eval_17.avg, eval_17.std))
                print("Dice eval_avg_dsc18: {:.5f} +- {:.3f}".format(eval_18.avg, eval_18.std))
                print("Dice eval_avg_dsc19: {:.5f} +- {:.3f}".format(eval_19.avg, eval_19.std))
                print("Dice eval_avg_dsc20: {:.5f} +- {:.3f}".format(eval_20.avg, eval_20.std))
                print("Dice eval_avg_dsc21: {:.5f} +- {:.3f}".format(eval_21.avg, eval_21.std))
                print("Dice eval_avg_dsc22: {:.5f} +- {:.3f}".format(eval_22.avg, eval_22.std))


            if save_validation_img:

                if_name = save_result_dir + str(eval_idx)+ '_moved_label.nii'
                gt_name = save_result_dir + str(eval_idx)+ '_fixed_label.nii'

                save_img(out_seg.float(), if_name, 'label') 
                save_img(y_seg.float(), gt_name, 'label') 

                if_name = save_result_dir + str(eval_idx)+ '_moved_img.nii.gz'
                gt_name = save_result_dir + str(eval_idx)+ '_fixed_img.nii.gz'

                save_img(out.float(), if_name, 'img') 
                save_img(y.float(), gt_name, 'img') 

                if_name = save_result_dir + str(eval_idx)+ '_moving_img.nii.gz'
                gt_name = save_result_dir + str(eval_idx)+ '_moving_label.nii.gz'

                save_img(data[4].float(), if_name, 'img') 
                save_img(data[6].float(), gt_name, 'label') 

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--affine_model', type=str, default='experiments/affine/',
                        help='Affine model load directory')
    parser.add_argument('--dir_model', type=str, default='experiments/test/',
                        help='DIR model load directory')
    parser.add_argument('--Dataset', type=str, default='Dataset/Segrap2023/',
                        help='Dataset directory')
    parser.add_argument('--save_validation_img', type=str2bool, default='False',
                        help='save_validation_img True or False')
    parser.add_argument('--img_size', type=int, default=(112, 144, 320),
                        help='size of image')
    parser.add_argument('--patch_size', type=int, default=(64, 80, 160),
                        help='size of patch')
    parser.add_argument('--sliding_stride', type=int, default=(46, 64, 80), 
                        help='sliding_stride')
    
    args = parser.parse_args()
    main(args)