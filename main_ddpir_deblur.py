import os.path
import cv2
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict
import hdf5storage
from scipy import ndimage

import torch

from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils import utils_model

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
"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img         = 2.55/255.0           # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'diffusion_ffhq_10m'  #diffusion_ffhq_10m, drunet_color, set diffusino model
    testset_name            = 'set5'            # set testing set,  'set18' | 'set24'
    iter_num                = 1000              # set number of iterations, default: 40 for demosaicing
    iter_num_U              = 1                 # set number of inner iterations, default: 1

    show_img                = False             # default: False
    save_L                  = True             # save LR image
    save_E                  = False             # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = True              # save generation process
    border                  = 0
    
    t_start                 = 999               # start timestep of the diffusion process

    sigma                   = max(0.01,noise_level_img)  # noise level associated with condition y
    lambda_                 = 1.                # key parameter lambda
    sub_1_analytic          = True              # use analytical solution
    
    ddim_sample             = False             # sampling method
    model_out_type          = 'pred_xstart'     # model output type: pred_x_prev; pred_xstart; epsilon; score
    skip_type               = 'uniform'         # uniform, quad
    eta                     = 1.                # eta for ddim sampling
    
    sf                      = 1
    task_current            = 'deblur'          
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


    # --------------------------------
    # load kernel
    # --------------------------------

    kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'Levin09.mat'))['kernels']

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    if 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
    elif 'diffusion' in model_name:
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

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []  # record average PSNR for each kernel
    
    noise_model_t = find_nearest(reduced_alpha_cumprod,noise_level_model)

    for k_index in range(kernels.shape[1]):

        logger.info('-------k:{:>2d} ---------'.format(k_index))
        test_results = OrderedDict()
        test_results['psnr'] = []
        k = kernels[0, k_index].astype(np.float64)
        util.imshow(k) if show_img else None

        def test_rho(lambda_=lambda_, model_out_type=model_out_type):
            for idx, img in enumerate(L_paths):

                # --------------------------------
                # (1) get img_L
                # --------------------------------

                img_name, ext = os.path.splitext(os.path.basename(img))
                img_H = util.imread_uint(img, n_channels=n_channels)
                img_H = util.modcrop(img_H, 8)  # modcrop

                img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
                util.imshow(img_L) if show_img else None
                img_L = util.uint2single(img_L)

                np.random.seed(seed=0)  # for reproducibility
                img_L = img_L * 2 - 1
                img_L += np.random.normal(0, noise_level_img * 2, img_L.shape) # add AWGN
                img_L = img_L / 2 + 0.5

                # --------------------------------
                # (2) get rhos and sigmas
                # --------------------------------

                sigmas = []
                sigma_ks = []
                rhos = []
                for i in range(num_train_timesteps):
                    sigmas.append(reduced_alpha_cumprod[999-i])
                    if model_out_type == 'pred_xstart':
                        sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                    elif model_out_type == 'pred_x_prev':
                        sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                    rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))
                        
                rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)
                
                # --------------------------------
                # (3) initialize x, and pre-calculation
                # --------------------------------

                # x = util.single2tensor4(img_L).to(device)

                y = util.single2tensor4(img_L).to(device)   #(1,3,256,256)
                y = y * 2 -1        # [-1,1]

                t_y = find_nearest(reduced_alpha_cumprod,noise_level_img)
                sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
                x = sqrt_alpha_effective * y + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
                        sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)
                # x = torch.randn_like(y)     

                k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)

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
                    # skip iters
                    if t_i > t_start:
                        continue
                    for u in range(iter_num_U):
                        # --------------------------------
                        # step 1, reverse diffsuion step
                        # --------------------------------

                        if 'drunet' in model_name:
                            x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                            x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)

                        elif 'diffusion' in model_name:
                            # solve equation 6b with one reverse diffusion step
                            if skip > 1 and model_out_type == 'pred_x_prev' and i != len(seq)-1:
                                # generalized ddim sampling method
                                t_im1 = find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                                x0 = model_fn(x, noise_level=curr_sigma*255,model_out_type='pred_xstart')
                                # x = ((torch.sqrt(alphas_cumprod[t_im1]) * betas[t_i]) * x0 + \
                                #     (torch.sqrt(alphas[t_i]) * sqrt_1m_alphas_cumprod[t_im1]**2) * x) \
                                #          / (1.0 - alphas_cumprod[t_i])
                                # x = x + sqrt_1m_alphas_cumprod[t_im1]**2 / sqrt_1m_alphas_cumprod[t_i]**2 * betas[t_i] * torch.randn_like(x)
                                alpha_prod_t = alphas_cumprod[int(t_i)]
                                beta_prod_t = 1 - alpha_prod_t
                                eps = (x - alpha_prod_t ** (0.5) * x0) / beta_prod_t ** (0.5)
                                eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                                x = torch.sqrt(alphas_cumprod[t_im1]) * x0 + torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                            + eta_sigma * torch.randn_like(x)
                            else: 
                                x = model_fn(x, noise_level=curr_sigma*255,model_out_type=model_out_type)
                                x0 = x
                            # x = utils_model.test_mode(model_fn, x, mode=0, refield=32, min_size=256, modulo=16, vec_t=vec_t)

                        # --------------------------------
                        # step 2, FFT
                        # --------------------------------

                        # add noise, make the condition image noise level consistent in pixel level
                        if model_out_type == 'pred_xstart':
                            y_t = y
                        elif model_out_type == 'pred_x_prev':
                            y_t = sqrt_alphas_cumprod[t_i] * (y) + sqrt_1m_alphas_cumprod[t_i] * (torch.randn_like(y))
                            # y_t = y
                        if seq[i] != seq[-1]:
                            if sub_1_analytic:
                                y_t = y_t / 2 + 0.5         # [0,1]
                                FB, FBC, F2B, FBFy = sr.pre_calculate(y_t, k_tensor, sf)
                                tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                # when noise level less than given image noise, skip
                                if i < num_train_timesteps-noise_model_t:
                                    x = x / 2 + 0.5
                                    x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)
                                    x = x * 2 - 1
                                else:
                                    model_out_type = 'pred_x_prev'
                                    pass
                            else:
                                #Tx = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
                                #x = x - 1 / (2*rhos[t_i]) * (Tx - y_t)
                                pass

                        if (model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == iter_num_U-1):
                            if i < num_train_timesteps-noise_model_t: 
                                x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                            # for the last step without analytic solution, need to add noise with the help of model output
                            elif i == num_train_timesteps-noise_model_t: 
                                # generalized ddim sampling method
                                t_im1 = find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                                alpha_prod_t = alphas_cumprod[int(t_i)]
                                beta_prod_t = 1 - alpha_prod_t
                                eps = (x - alpha_prod_t ** (0.5) * x0) / beta_prod_t ** (0.5)
                                eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                                x = torch.sqrt(alphas_cumprod[t_im1]) * x0 + torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                            + eta_sigma * torch.randn_like(x)
                            else:
                                pass
                            
                        # set back to x_t from x_{t-1}
                        if u < iter_num_U-1 and seq[i] != seq[-1]:
                            x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)

                    # save the process
                    if save_progressive and (seq[i] in progress_seq):
                        x_0 = (x/2+0.5)
                        x_show = x_0.clone().detach().cpu().numpy()       #[0,1]
                        x_show = np.squeeze(x_show)
                        if x_show.ndim == 3:
                            x_show = np.transpose(x_show, (1, 2, 0))
                        progress_img.append(x_show)
                        logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                        
                        if show_img:
                            util.imshow(x_show)


                # --------------------------------
                # (3) img_E
                # --------------------------------

                img_E = util.tensor2uint(x_0)

                psnr = util.calculate_psnr(img_E, img_H, border=border)  # change with your own border
                test_results['psnr'].append(psnr)
                logger.info('{:->4d}--> {:>10s} --k:{:>2d} PSNR: {:.2f}dB'.format(idx+1, img_name+ext, k_index, psnr))

                if n_channels == 1:
                    img_H = img_H.squeeze()

                if save_E:
                    util.imsave(img_E, os.path.join(E_path, img_name+'_k'+str(k_index)+'_'+model_name+'.png'))

                if save_progressive:
                    now = datetime.now()
                    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                    img_total = cv2.hconcat(progress_img)
                    if show_img:
                        util.imshow(img_total,figsize=(80,4))
                    util.imsave(img_total*255., os.path.join(E_path, img_name+f'_process_lambda_k_{lambda_}_{current_time}_psnr_{psnr}.png'))
                    
                # --------------------------------
                # (4) img_LEH
                # --------------------------------

                if save_LEH:
                    img_L = util.single2uint(img_L)
                    k_v = k/np.max(k)*1.0
                    k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                    k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                    img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                    util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if show_img else None
                    util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_k'+str(k_index)+'_LEH.png'))

                if save_L:
                    util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_k'+str(k_index)+'_LR.png'))

        # experiments
        lambdas = [0.1*i for i in range(2,30)]
        for lambda_ in lambdas:
            test_rho(lambda_)

        # test with the first image in the path
        break

        # --------------------------------
        # Average PSNR
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), kernel: ({}) sigma: ({:.2f}): {:.2f} dB'.format(testset_name, k_index, noise_level_model, ave_psnr))
        test_results_ave['psnr'].append(ave_psnr)

if __name__ == '__main__':

    main()
