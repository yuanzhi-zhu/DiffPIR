import os.path
import logging
from re import T

import cv2
import torch
import numpy as np
from datetime import datetime

from utils import utils_logger
from utils import utils_image as util

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import argparse

"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_gray          # model_name, for color images
   |--drunet_color
|--testset                 # testsets
|--results                 # results
# --------------------------------------------

How to run:
step 1: download [drunet_color.pth, ircnn_color.pth] from https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
step 2: set your own testset 'testset_name' and parameter setting such as 'noise_level_model', 'iter_num'. 
step 3: 'python main_dpir_demosaick.py'

"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img         = 0/255.0           # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'diffusion_ffhq_10m'  # set diffusino model
    testset_name            = 'gts/face'        # set testing set,  'set18' | 'set24'
    mask_name               = 'gt_keep_masks/face/000000.png'
    iter_num                = 1000              # set number of iterations, default: 40 for demosaicing
    iter_num_U              = 1                 # set number of inner iterations, default: 1

    show_img                = False             # default: False
    save_L                  = False             # save LR image
    save_E                  = False             # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = True              # save generation process
    save_progressive_mask   = False             # save generation process

    sigma                   = max(0.01,noise_level_img)  # noise level associated with condition y
    lambda_                 = 1.                # key parameter lambda
    sub_1_analytic          = True              # use analytical solution
    
    model_out_type          = 'pred_xstart'     # pred_x_prev; pred_xstart; epsilon; score
    generate_mode           = 'DPIR'            # model output type: pred_x_prev; pred_xstart; epsilon; score
    skip_type               = 'uniform'         # uniform, quad
    ddim_sample             = False             # sampling method
    
    log_process             = False
    task_current            = 'ip'              # 'ip' for inpainting
    n_channels              = 3                 # fixed
    cwd                     = '/cluster/work/cvl/jinliang/ckpts_yuazhu/DDPIR/'
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results')      # fixed
    result_name             = testset_name + '_' + task_current + '_' + model_name
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # noise schedule 
    beta_start              = 0.002 / 1000
    beta_end                = 20 / 1000
    num_train_timesteps     = 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    # ----------------------------------------
    # L_path, E_path, H_path, mask_path
    # ----------------------------------------

    L_path                  = os.path.join(testsets, testset_name)      # L_path, for Low-quality images
    E_path                  = os.path.join(results, result_name)        # E_path, for Estimated images
    mask_path               = os.path.join(testsets, mask_name)         # mask_path, for mask images
    util.mkdir(E_path)

    logger_name             = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger                  = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    def create_argparser():
        defaults = dict(
            clip_denoised=True,
            num_samples=1,
            batch_size=1,
            use_ddim=False,
            model_path=model_path,
            diffusion_steps=num_train_timesteps,
            noise_schedule='linear',
            num_head_channels=64,
            resblock_updown=True,
            use_fp16=False,
            use_scale_shift_norm=True,
            num_heads=4,
            num_heads_upsample=-1,
            use_new_attention_order=False,
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            channel_mult="",
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=False,
            image_size=256,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
            dropout=0.1,
        )
        # defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser
    args = create_argparser().parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    # ----------------------------------------
    # wrap diffusion model
    # ----------------------------------------

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def model_fn(x, noise_level, vec_t=None, model_out_type=model_out_type, **model_kwargs):
        # time step corresponding to noise level
        if not torch.is_tensor(vec_t):
            t_step = find_nearest(reduced_alpha_cumprod,(noise_level/255.))
            vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)
            # timesteps = torch.linspace(1, 1e-3, num_train_timesteps, device=device)
            # t = timesteps[t_step]
        if not ddim_sample:
            out = diffusion.p_sample(
                model,
                x,
                vec_t,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )
        else:
            out = diffusion.ddim_sample(
                model,
                x,
                vec_t,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=model_kwargs,
                eta=0,
            )
        if model_out_type == 'pred_x_prev_and_start':
            return out["sample"], out["pred_xstart"]
        elif model_out_type == 'pred_x_prev':
            out = out["sample"]
        elif model_out_type == 'pred_xstart':
            out = out["pred_xstart"]
        elif model_out_type == 'epsilon':
            alpha_prod_t = alphas_cumprod[int(t_step)]
            beta_prod_t = 1 - alpha_prod_t
            out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
        elif model_out_type == 'score':
            alpha_prod_t = alphas_cumprod[int(t_step)]
            beta_prod_t = 1 - alpha_prod_t
            out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
            out = - out / beta_prod_t ** (0.5)
                
        return out

    def test_rho(lambda_=lambda_):
        for idx, img in enumerate(L_paths):

            # --------------------------------
            # (1) get img_H and img_L
            # --------------------------------

            idx += 1
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)

            # --------------------------------
            # (2) initialize x
            # --------------------------------

            mask = util.imread_uint(mask_path, n_channels=n_channels).astype(bool)

            img_L = img_H * mask    #(256,256,3)
            y = util.uint2tensor4(img_L).to(device)   #(1,3,256,256)
            y = y * 2 -1        # [-1,1]

            mask = util.single2tensor4(mask.astype(np.float32)).to(device) 

            x = torch.randn_like(y)        

            # --------------------------------
            # (3) get rhos and sigmas
            # --------------------------------

            sigmas = []
            sigma_ks = []
            rhos = []
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[999-i])
                sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))
                     
            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)

            # --------------------------------
            # (4) main iterations
            # --------------------------------

            progress_img = []
            # create sequence of timestep for sampling
            skip = num_train_timesteps//iter_num
            if skip_type == 'uniform':
                seq = [i*skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps-1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::(len(seq)//10)]
            progress_seq.append(seq[-1])
            # reverse diffusion for one image from random noise
            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                # time step associated with the noise level sigmas[i]
                t_i = find_nearest(reduced_alpha_cumprod,curr_sigma)
                #vec_t = torch.tensor([999-i] * x.shape[0], device=device)

                for u in range(iter_num_U):
                    # --------------------------------
                    # step 1, reverse diffsuion step
                    # --------------------------------

                    # add noise, make the image noise level consistent in pixel level
                    if generate_mode == 'repaint':
                        x = (sqrt_alphas_cumprod[t_i] * y + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                                + (1-mask) * x

                    # solve equation 6b with one reverse diffusion step
                    x = model_fn(x, noise_level=curr_sigma*255,model_out_type=model_out_type)

                    # x = utils_model.test_mode(model_fn, x, mode=0, refield=32, min_size=256, modulo=16, noise_level=sigmas[i].cpu().numpy()*255)
                    # --------------------------------
                    # step 2, closed-form solution
                    # --------------------------------

                    # analytic solution to equation 6a
                    if generate_mode == 'DPIR': 
                        # solve sub-problem
                        if sub_1_analytic:
                            x = (mask*y + rhos[t_i].float()*x).div(mask+rhos[t_i])
                        else:
                            # TODO: first order solver
                            # x = x - 1 / (2*rhos[t_i]) * (x - y_t) * mask 
                            pass

                        if (model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == iter_num_U-1):
                            x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                        
                    # set back to x_t from x_{t-1}
                    if u < iter_num_U-1 and seq[i] != seq[-1]:
                        x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)

                # save the process
                if save_progressive and (seq[i] in progress_seq):
                    x_show = (x/2+0.5).clone().detach().cpu().numpy()       #[0,1]
                    x_show = np.squeeze(x_show)
                    if x_show.ndim == 3:
                        x_show = np.transpose(x_show, (1, 2, 0))
                    progress_img.append(x_show)
                    if log_process:
                        logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                    if show_img:
                        util.imshow(x_show)

            # recover conditional part
            if generate_mode in ['repaint','DPIR']:
                x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

            # --------------------------------
            # (4) save process
            # --------------------------------
            
            img_E = util.tensor2uint(x)

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

            if save_L:
                util.imsave(img_L, os.path.join(E_path, img_name+'_L.png'))

            if save_LEH:
                util.imsave(np.concatenate([img_L, img_E, img_H], axis=1), os.path.join(E_path, img_name+model_name+'_LEH.png'))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                if generate_mode in ['repaint','DPIR']:
                    mask = np.squeeze(mask.cpu().numpy())
                    if mask.ndim == 3:
                        mask = np.transpose(mask, (1, 2, 0))
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total,figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+f'_process_lambda_{lambda_}_{current_time}.png'))
                images = []
                y_t = np.squeeze((y/2+0.5).cpu().numpy())
                if y_t.ndim == 3:
                    y_t = np.transpose(y_t, (1, 2, 0))
                if generate_mode in ['repaint','DPIR']:
                    for x in progress_img:
                        images.append((y_t)* mask+ (1-mask) * x)
                    img_total = cv2.hconcat(images)
                    if show_img:
                        util.imshow(img_total,figsize=(80,4))
                    if save_progressive_mask:
                        util.imsave(img_total*255., os.path.join(E_path, img_name+f'_process_mask_lambda_k_{lambda_}_{current_time}.png'))

            # test with the first image in the path
            break

    # experiments
    lambdas = [0.1*i for i in range(10,30)]
    for lambda_ in lambdas:
        test_rho(lambda_)


if __name__ == '__main__':

    main()
