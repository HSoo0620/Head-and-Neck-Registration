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
from models.TransMorph_Origin_Affine import CONFIGS as AFF_CONFIGS_TM
import models.TransMorph_Origin_Affine as TransMorph_affine
from torch.autograd import Variable
from einops.einops import rearrange
import torch.nn.functional as F
import gc

from HnN_F import make_Data_case, make_gaussian_kernel, save_img, crop_img_HnN
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
    affine_model.load_state_dict(torch.load(affine_model_dir + natsorted(os.listdir(affine_model_dir), reverse=True)[0])['state_dict'])
    print('Affine Model: {} loaded!'.format(natsorted(os.listdir(affine_model_dir), reverse=True)[0]))
    affine_model.cuda()
    affine_model.eval()
    for param in affine_model.parameters():
        param.requires_grad_(False)

    AffInfer_near = TransMorph_affine.ApplyAffine(mode='nearest')
    AffInfer_near.cuda()
    AffInfer_bi = TransMorph_affine.ApplyAffine(mode='bilinear')
    AffInfer_bi.cuda()

    # Loss function weights
    dir_weights = [0.5, 0.5, 0.5, 1.0]

    # Save Directory
    save_dir = args.dir_model
    save_result_dir = 'results/' + save_dir
    if not os.path.exists(save_result_dir):
           os.makedirs(save_result_dir)

    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)

    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    
    sys.stdout = Logger('logs/'+save_dir)


    '''hyper parameters'''
    lr = args.learning_rate 
    batch_size = args.batch_size
    epoch_start = 0
    max_epoch = args.max_epoch
    add_img_to_tensorboard = args.add_img_tensorboard
    validation_iter = args.validation_iter
    patch_size = args.patch_size # d h w patch size
    patch_size_gaussian = (64j, 80j, 160j) # d h w patch size
    img_size = args.img_size # d h w img size
    save_validation_img = args.save_validation_img
    sliding_stride = args.sliding_stride

    patch_z_size, patch_y_size, patch_x_size = patch_size
    img_z_size, img_y_size, img_x_size = img_size 

    # Use Pre-trained Model  
    cont_training_Basic = args.pre_train
    '''
    Initialize model
    '''
    config = CONFIGS_TM['M-Adv']
    model = M_Adv.M_Adv_model(config)
    model.cuda()

    '''
    If continue from previous training
    '''
    print("cont_training_Basic: ",cont_training_Basic)
    if cont_training_Basic:
        model_dir = args.dir_model
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9) ,8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('M-Adv Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)

    else:
        updated_lr = lr
    
    '''
    Initialize spatial transformation function
    '''
    reg_model_bilin_origin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin_origin.cuda()
    reg_model_bilin_patch = utils.register_model(patch_size, 'bilinear')
    reg_model_bilin_patch.cuda()

    reg_model_origin = utils.register_model(img_size, 'nearest')
    reg_model_origin.cuda()
    reg_model_patch = utils.register_model(patch_size, 'nearest')
    reg_model_patch.cuda()

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

    '''
    Head and Neck Dataset Load for Train & Valid
    '''
    img_name = args.Dataset + 'train/image/'
    label_name = args.Dataset + 'train/label/'
    train_pairs = make_Data_case(img_name, label_name)
    print("===Make Train Case : ", len(train_pairs), " Combinations")

    img_name = args.Dataset + 'test/image/'
    label_name = args.Dataset + 'test/label/'
    valid_pairs = make_Data_case(img_name, label_name)
    print("===Make Test Case : ", len(valid_pairs), " Combinations")

    train_set = datasets.HnNDataset_train(train_pairs, transforms=train_composed, patch_size=args.patch_size)
    val_set = datasets.HnNDataset_valid(valid_pairs, transforms=val_composed, patch_size=args.patch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)

    criterion_dsc = losses.Dice(bg=1)
    criterion_ncc = losses.NCC_HnN(win=args.patch_size)
    criterion_det = losses.Determ()
    criterion_reg = losses.Grad3d(penalty='l2')
    criterion_avg_dsc = losses.Dice_avg(bg=1)

    gaussian_kernel = make_gaussian_kernel(patch_size_gaussian)

    writer_moved = SummaryWriter(log_dir='logs/'+save_dir+'moved_and_fixed')
    writer_dff_label = SummaryWriter(log_dir='logs/'+save_dir+'m_f_dff_label')
    writer = SummaryWriter(log_dir='logs/'+save_dir)

    best_dsc = 0

    for epoch in range(epoch_start, max_epoch):
        torch.autograd.set_detect_anomaly(True)
        print('Training Start')
        print('Epoch {} :'.format(epoch))
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        train_DSC = utils.AverageMeter()

        idx = 0
        for data in train_loader:
            optimizer.zero_grad()
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            
            ''' affine Registration '''
            x_af = data[0]
            y_af = data[1]
            x_af_in = torch.cat((x_af, y_af), dim=1)
            with torch.no_grad():
                _, affine_mat, _, _, _, _ = affine_model(x_af_in)

            x = AffInfer_bi(data[4].float(), affine_mat)
            y = data[5].float()            
            x_seg = AffInfer_near(data[6].float(), affine_mat)
            y_seg = data[7].float()

            ''' patch crop '''
            x, y, x_seg, y_seg = crop_img_HnN(x, y, x_seg, y_seg, patch_size, img_size)

            ''' DIR Model Train '''
            model.train()
            x_in = torch.cat((x,y), dim=1)
            out, flow, _, _, _, _ = model(x_in)
            out_seg = reg_model_patch([x_seg.float(), flow.float()])
            
            ''' Calculate loss'''
            loss_ncc = criterion_ncc(out, y); loss_ncc_w = loss_ncc * dir_weights[0]
            loss_dsc = criterion_dsc(out_seg, y_seg); loss_dsc_w = loss_dsc * dir_weights[1]
            loss_reg = criterion_reg(flow); loss_reg_w = loss_reg * dir_weights[2]
            loss_det = criterion_det(flow); loss_det_w = loss_det * dir_weights[3]
            loss_dsc_avg,_ = criterion_avg_dsc(out_seg, y_seg)
            loss_dsc_avg_w = loss_dsc_avg * dir_weights[1]

            dir_loss = loss_ncc_w + loss_dsc_w + loss_reg_w + loss_det_w +loss_dsc_avg_w
            loss = dir_loss
            loss_all.update(loss.item(), x.numel())
            train_DSC.update(1-loss_dsc.item(), y.numel())
            
            loss.backward() 
            optimizer.step() 

            writer.add_scalar('Loss/train', loss_all.val, idx+epoch)
            writer.add_scalar('DSC/train', train_DSC.val, idx+epoch)

            if add_img_to_tensorboard:
                if idx % 10 == 0:            
                    x = rearrange(x.clone().detach(), 'a b d h w ->a b w h d ')
                    y = rearrange(y.clone().detach(), 'a b d h w ->a b w h d ')
                    out = rearrange(out.clone().detach(), 'a b d h w ->a b w h d ')   
                    slice_num = 31           
                    
                    writer_moved.add_image('train',torch.cat((out[0][0][:,:,slice_num], y[0][0][:,:,slice_num], torch.abs(y[0][0][:,:,slice_num]-out[0][0][:,:,slice_num])), dim=0) , idx+epoch, dataformats='WC')
                    y_seg = torch.argmax(y_seg, dim=1).float()
                    out_seg = reg_model_patch([x_seg.float(), flow.float()])
                    out_seg = torch.argmax(out_seg, dim=1).float()

                    y_seg = rearrange(y_seg.clone().detach(), 'a d h w ->a w h d ')
                    out_seg = rearrange(out_seg.clone().detach(), 'a d h w ->a w h d ')
                    writer_dff_label.add_image('train',torch.cat((out_seg[0][:,:,slice_num], y_seg[0][:,:,slice_num], torch.abs(y_seg[0][:,:,slice_num]-out_seg[0][:,:,slice_num])), dim=0), idx+epoch, dataformats='WC')

            del x, y, x_seg, y_seg, out_seg
            print('Iter {} of {} loss {:.4f}, NCC: {:.4f}, DSC: {:.4f}, avg_DSC: {:.4f}, REG: {:.6f}, DET: {:.6f}, lr: {:.5f}'.format(idx, len(train_loader),
                                                                            loss.item(),
                                                                            loss_ncc.item(),
                                                                            1-loss_dsc.item(),
                                                                            1-loss_dsc_avg.item(),
                                                                            loss_reg.item(),
                                                                            loss_det.item(),
                                                                            updated_lr))
            '''
            Validation
            '''

            if idx % validation_iter == 0:
                print('Validation Start')
                eval_dsc = utils.AverageMeter()
                eval_avg_dsc = utils.AverageMeter()
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

                        total_DVF = torch.zeros(img_z_size, img_y_size, img_x_size ,3)
                        total_divid = torch.zeros(img_z_size, img_y_size, img_x_size ,3)
                        dvf_num=0

                        for d in range(2):
                            for h in range(2):
                                for w in range(3):
                                    dvf_num += 1
                                    sum_divid = torch.ones((patch_z_size, patch_y_size, patch_x_size , 3))

                                    x_buf_img = x[:,:,d*sliding_stride[0]:d*sliding_stride[0]+patch_z_size, h*sliding_stride[1]:h*sliding_stride[1]+patch_y_size, w*sliding_stride[2]:w*sliding_stride[2]+patch_x_size]
                                    y_buf_img = y[:,:,d*sliding_stride[0]:d*sliding_stride[0]+patch_z_size, h*sliding_stride[1]:h*sliding_stride[1]+patch_y_size, w*sliding_stride[2]:w*sliding_stride[2]+patch_x_size]

                                    x_in = torch.cat((x_buf_img.cuda(), y_buf_img.cuda()), dim=1)
                                    _, flow, _, _, _, _ = model(x_in)

                                    buf_dvf = rearrange(flow[0].cpu().detach(), 'c d h w ->d h w c')
                                    total_DVF[d*sliding_stride[0]:d*sliding_stride[0]+patch_z_size, h*sliding_stride[1]:h*sliding_stride[1]+patch_y_size, w*sliding_stride[2]:w*sliding_stride[2]+patch_x_size,:]+= buf_dvf * gaussian_kernel
                                    total_divid[d*sliding_stride[0]:d*sliding_stride[0]+patch_z_size, h*sliding_stride[1]:h*sliding_stride[1]+patch_y_size, w*sliding_stride[2]:w*sliding_stride[2]+patch_x_size,:] += sum_divid * gaussian_kernel

                                    del x_buf_img, y_buf_img, x_in, flow, buf_dvf, _
                                    
                        total_div_result = total_DVF / total_divid
                        total_div_result = rearrange(total_div_result[None], 'b d h w c -> b c d h w')

                        out_seg = reg_model_origin([x_seg.float(), total_div_result.float()])
                        out = reg_model_bilin_origin([x.float(), total_div_result.float()])
                        
                        dsc = 1-criterion_dsc(out_seg, y_seg)
                        eval_dsc.update(dsc.item(), x.size(0))

                        avg_dsc, organs_dsc = criterion_avg_dsc(out_seg, y_seg)
                        avg_dsc = 1-avg_dsc
                        eval_avg_dsc.update(avg_dsc.item(), x.size(0))

                        print('Idx {} of Val {} DSC:{: .4f}'.format(eval_idx, len(val_loader),dsc.item()))
                        if eval_idx == 30:
                            print("--everage Dice eval_dsc: {:.5f} +- {:.3f}".format(eval_dsc.avg, eval_dsc.std))
                            print("--everage Dice eval_avg_dsc: {:.5f} +- {:.3f}".format(eval_avg_dsc.avg, eval_dsc.std))

                        if save_validation_img:
                            if_name = save_result_dir + str(eval_idx)+ '_moved_mask.nii.gz'
                            gt_name = save_result_dir + str(eval_idx)+ '_fixed_mask.nii.gz'

                            save_img(out_seg.float(), if_name, 'label') 
                            save_img(y_seg.float(), gt_name, 'label') 

                            if_name = save_result_dir + str(eval_idx)+ '_moved_img.nii.gz'
                            gt_name = save_result_dir + str(eval_idx)+ '_fixed_img.nii.gz'

                            save_img(out.float(), if_name, 'img') 
                            save_img(y.float(), gt_name, 'img') 

                            
                            if_name = save_result_dir + str(eval_idx)+ '_moving_img.nii.gz'
                            gt_name = save_result_dir + str(eval_idx)+ '_moving_mask.nii.gz'

                            save_img(data[4].float(), if_name, 'img') 
                            save_img(data[6].float(), gt_name, 'label') 
                            
                            y_seg = torch.argmax(y_seg, dim=1).float()
                            out_seg = torch.argmax(out_seg, dim=1).float()

                        torch.cuda.empty_cache()
                        gc.collect()
                best_dsc = max(eval_dsc.avg, best_dsc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                }, save_dir='experiments/'+save_dir, filename='dsc{:.4f} idx{:.1f} epoch{:.1f}.pth.tar'.format(eval_dsc.avg,idx,epoch))

                writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)                                        
                loss_all.reset()


    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
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

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('GPU #' + str(GPU_idx) + ': ' + GPU_name)
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
    parser.add_argument('--pre_train', type=str2bool, default='False',
                        help='pre-train load True or False')
    parser.add_argument('--add_img_tensorboard', type=str2bool, default='False',
                        help='add img tensorboard True or False')
    parser.add_argument('--save_validation_img', type=str2bool, default='False',
                        help='save_validation_img True or False')
    parser.add_argument('--img_size', type=int, default=(112, 144, 320),
                        help='size of image')
    parser.add_argument('--patch_size', type=int, default=(64, 80, 160),
                        help='size of patch')
    parser.add_argument('--sliding_stride', type=int, default=(46, 64, 80), 
                        help='sliding_stride')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='max_epoch')
    parser.add_argument('--validation_iter', type=int, default=1000,
                        help='validation_iter')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')                        
    parser.add_argument('--learning_rate', type=int, default=1e-4,
                        help='learning_rate')       

    args = parser.parse_args()
    main(args)