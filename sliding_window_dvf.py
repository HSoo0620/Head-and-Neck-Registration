import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn.functional as F
import nibabel as nib
from einops.einops import rearrange
from dcm2tensor import resizing_img

def forward_DIR(model, affine_out, fixed_img, dir_d,dir_h,dir_w, step_d, step_h, step_w, patch_d, patch_h, patch_w) :
    total_DVF = torch.zeros(dir_d,dir_h,dir_w,3).cuda()
    total_divid = torch.zeros(dir_d,dir_h,dir_w,3).cuda()
    gaussian_filter = make_gaussian_filter() # gaussian filter
    sum_divid = torch.ones((patch_d, patch_h, patch_w, 3)).cuda()
    dvf_num = 0
    for d in range(2):
        for h in range(2):
            for w in range(3):
                dvf_num += 1

                x_buf_img = affine_out[:,:,d*step_d:d*step_d+patch_d, h*step_h:h*step_h+patch_h, w*step_w:w*step_w+patch_w]
                y_buf_img = fixed_img[:,:,d*step_d:d*step_d+patch_d, h*step_h:h*step_h+patch_h, w*step_w:w*step_w+patch_w]

                x_in = torch.cat((x_buf_img.cuda(), y_buf_img.cuda()), dim=1)
                _, flow, _, _, _, _ = model(x_in)

                flow = rescale_dvf(flow, patch_d, patch_h, patch_w)
                buf_dvf = rearrange(flow[0], 'c d h w ->d h w c').cuda()
                total_DVF[d*step_d:d*step_d+patch_d, h*step_h:h*step_h+patch_h, w*step_w:w*step_w+patch_w,:]+= buf_dvf * gaussian_filter
                total_divid[d*step_d:d*step_d+patch_d, h*step_h:h*step_h+patch_h, w*step_w:w*step_w+patch_w,:] += sum_divid * gaussian_filter
    return total_DVF, total_divid

def restore_DVF(flow, resize_d, resize_h, resize_w, crop_d, crop_h, crop_w, original_d, original_h, original_w, bbox, fixed_d, fixed_h, fixed_w, fixed_bbox):
    """
    flow : (B C resized_d resied_h resized_w) shape의 DVF
    original_shape

    fixed로 padding후, moving으로 resize 
    """
    ### 2-1. flow resizing
    flow = flow[0]
    print(bbox)
    sclae_m_w, sclae_m_h, sclae_m_d = (bbox[0][1] - bbox[0][0]) / original_w, (bbox[1][1] - bbox[1][0]) / original_h, (bbox[2][1] - bbox[2][0]) / original_d
    sclae_f_w, sclae_f_h, sclae_f_d = (fixed_bbox[0][1] - fixed_bbox[0][0]) / fixed_w, (fixed_bbox[1][1] - fixed_bbox[1][0]) / fixed_h, (fixed_bbox[2][1] - fixed_bbox[2][0]) / fixed_d
    scale_w, scale_h, scale_d = sclae_f_w / sclae_m_w, sclae_f_h / sclae_m_h, sclae_f_d / sclae_m_d
    
    print(scale_w, scale_h, scale_d)

    upsampling_flow_z = resizing_img(flow[0].cpu()[None], mode='bilinear', depth=resize_d, height=resize_h, width=resize_w)
    upsampling_flow_y = resizing_img(flow[1].cpu()[None], mode='bilinear', depth=resize_d, height=resize_h, width=resize_w)
    upsampling_flow_x = resizing_img(flow[2].cpu()[None], mode='bilinear', depth=resize_d, height=resize_h, width=resize_w)
    ### 2-2. flow rescaling
    upsampling_flow_z *= (resize_d / crop_d)
    upsampling_flow_y *= (resize_h / crop_h)
    upsampling_flow_x *= (resize_w / crop_w)
    crop_dvf = torch.cat([upsampling_flow_z, upsampling_flow_y, upsampling_flow_x], dim=1) # (c(z,y,x), w, h, d)

    neg_w, neg_h, neg_d = bbox[0][0], bbox[1][0], bbox[2][0]
    pos_w, pos_h, pos_d = bbox[0][1], bbox[1][1], bbox[2][1]

    restored_dvf = torch.zeros(1, 3, original_d, original_h, original_w).cuda()
    restored_dvf[:, :, neg_d:pos_d, neg_h:pos_h, neg_w:pos_w] = crop_dvf

    ### 2-3. 512 resolution upsampling dvf
    return restored_dvf, crop_dvf, scale_w, scale_h, scale_d


def make_gaussian_filter():
    z, y, x = np.mgrid[-1.0:1.0:64j, -1.0:1.0:80j, -1.0:1.0:160j]
    zyx = np.column_stack([z.flat, y.flat, x.flat])
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([.25, .25, .25])
    covariance = np.diag(sigma**2)
    gaussian_kernel = multivariate_normal.pdf(zyx, mean=mu, cov=covariance)
    gaussian_kernel = gaussian_kernel.reshape(x.shape)
    gaussian_kernel = (gaussian_kernel - gaussian_kernel.min()) / (gaussian_kernel.max() - gaussian_kernel.min()) + 0.0001 # 1.0001 ~ 0.0001
    gaussian_kernel = torch.Tensor(gaussian_kernel)
    gaussian_kernel = torch.stack([gaussian_kernel, gaussian_kernel, gaussian_kernel], dim=-1).cuda()

    return gaussian_kernel


def affine_to_dvf(transformation_mat, d, h, w) :
    identity_mat = torch.Tensor([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])[None]
    identity_mat_affine_grid = torch.nn.functional.affine_grid(identity_mat, [1, 3, d, h, w], align_corners=True)
    resized_affine_grid = torch.nn.functional.affine_grid(transformation_mat, [1, 3, d, h, w], align_corners=True)
    affine_dvf = resized_affine_grid - identity_mat_affine_grid.cuda()

    affine_dvf = affine_dvf * torch.Tensor([w, h, d]).cuda()
    affine_dvf = rearrange(affine_dvf, 'b d h w c -> b c d h w')[0]
    swapped_dvf = torch.stack([affine_dvf[2], affine_dvf[1], affine_dvf[0]], dim=0) # c(x y z -> z y x)
    return swapped_dvf


def rescale_and_resize_dvf(dvf, src_size, target_size) :
    output_dvf = resizing_img(dvf[0].cpu(), mode='bilinear', depth=target_size[0], height=target_size[1], width=target_size[2])
    output_dvf = rearrange(output_dvf, 'b c d h w -> b d h w c')[0]
    output_dvf = output_dvf * torch.Tensor([target_size[0] / src_size[0], target_size[1] / src_size[1], target_size[2] / src_size[2]])
    output_dvf = rearrange(output_dvf, 'd h w c -> c d h w')[None]
    return output_dvf


def resize_dvf(img, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() # (64, 128, 128, 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)
    img = img.cpu()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = img[0][0]
    return img


def rescale_dvf(flow, d, h, w):
    upsampling_flow_z = resize_dvf(flow[0][0], d, h, w)
    upsampling_flow_y = resize_dvf(flow[0][1], d, h, w)
    upsampling_flow_x = resize_dvf(flow[0][2], d, h, w)
    
    upsampling_flow_z *= (d / 64)
    upsampling_flow_y *= (h / 80)
    upsampling_flow_x *= (w / 160)
            
    upsampling_flow = torch.cat([upsampling_flow_z[None], upsampling_flow_y[None], upsampling_flow_x[None]], dim=0) # (c, w, h, d)
    upsampling_flow = upsampling_flow[None] # (1, c, w, h, d)
    return upsampling_flow


def HnN_resize_img(img, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() # (64, 128, 128, 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)
    img = img.cpu()
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    return img

    
def HnN_save_img(img, img_name) :
    img = img.squeeze() 
    img = img.cpu().numpy()
    img = img.swapaxes(0, 2)
    x = nib.Nifti1Image(img, None)
    nib.save(x, img_name)