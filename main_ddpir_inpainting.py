import os.path
import logging

import cv2
import torch
import numpy as np
from datetime import datetime
from collections import OrderedDict

from utils import utils_model
from utils import utils_logger
from utils import utils_image as util

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img         = 0/255.0           # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'diffusion_celeba256_250000'  # diffusion_celeba256_250000, diffusion_ffhq_10m; set diffusino model
    testset_name            = 'gts/face'        # set testing set, 'ffhq_val'
    mask_name               = 'gt_keep_masks/face/000000.png'
    num_train_timesteps     = 1000
    iter_num                = 1000              # set number of iterations, default: 40 for demosaicing
    iter_num_U              = 1                 # set number of inner iterations, default: 1
    skip                    = num_train_timesteps//iter_num     # skip interval

    show_img                = False             # default: False
    save_L                  = False             # save LR image
    save_E                  = False             # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = True              # save generation process
    save_progressive_mask   = False             # save generation process

    sigma                   = max(0.001,noise_level_img)  # noise level associated with condition y
    lambda_                 = 1.                # key parameter lambda
    sub_1_analytic          = True              # use analytical solution
    eta                     = 1.0                # eta for ddim samplingn  
    zeta                    = 1.0                      
    guidance_scale          = 1.0   
    
    model_out_type          = 'pred_xstart'     # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode           = 'DDPIR'           # repaint; vanilla; DDPIR
    skip_type               = 'uniform'         # uniform, quad
    ddim_sample             = False             # sampling method
    
    log_process             = False
    task_current            = 'ip'              # 'ip' for inpainting
    n_channels              = 3                 # fixed
    cwd                     = '/cluster/work/cvl/jinliang/ckpts_yuazhu/DDPIR/'
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results')      # fixed
    result_name             = f'{testset_name}_{task_current}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_zeta{zeta}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    calc_LPIPS              = False
    
    # noise schedule 
    beta_start              = 0.002 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0

    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
    t_start                 = 999   

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

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
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
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, skipstep analytic steps:{}'.format(eta, zeta, lambda_, noise_model_t))
    logger.info('start step:{}, skip_type:{}, skip interval:{}'.format(t_start, skip_type, skip))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    def test_rho(lambda_=lambda_):
        test_results = OrderedDict()
        if calc_LPIPS:
            import lpips
            loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
            test_results['lpips'] = []

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

            # x = torch.randn_like(y)
            x = sqrt_alphas_cumprod[t_start] * y + sqrt_1m_alphas_cumprod[t_start] * torch.randn_like(y)   

            # --------------------------------
            # (3) get rhos and sigmas
            # --------------------------------

            sigmas = []
            sigma_ks = []
            rhos = []
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart':
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                elif model_out_type == 'pred_x_prev':
                    sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))
                     
            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)

            # --------------------------------
            # (4) main iterations
            # --------------------------------

            progress_img = []
            # create sequence of timestep for sampling
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
                t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
                # skip iters
                if t_i > t_start:
                    continue
                for u in range(iter_num_U):
                    # --------------------------------
                    # step 1, reverse diffsuion step
                    # --------------------------------

                    # add noise, make the image noise level consistent in pixel level
                    if generate_mode == 'repaint':
                        x = (sqrt_alphas_cumprod[t_i] * y + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                                + (1-mask) * x

                    # solve equation 6b with one reverse diffusion step
                    if model_out_type == 'pred_xstart':
                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                    else:
                        x = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                    # x = utils_model.test_mode(model_fn, x, mode=0, refield=32, min_size=256, modulo=16, noise_level=sigmas[i].cpu().numpy()*255)
                    # --------------------------------
                    # step 2, closed-form solution
                    # --------------------------------

                    # analytic solution
                    if generate_mode == 'DDPIR': 
                        # solve sub-problem
                        if sub_1_analytic:
                            if model_out_type == 'pred_xstart':
                                # when noise level less than given image noise, skip
                                if i < num_train_timesteps-noise_model_t:    
                                    x0_p = (mask*y + rhos[t_i].float()*x0).div(mask+rhos[t_i])
                                    x0 = x0 + guidance_scale * (x0_p-x0)
                                else:
                                    pass
                            elif model_out_type == 'pred_x_prev':
                                # when noise level less than given image noise, skip
                                if i < num_train_timesteps-noise_model_t:    
                                    x = (mask*y + rhos[t_i].float()*x).div(mask+rhos[t_i]) # y-->yt ?
                                else:
                                    pass
                        else:
                            # TODO: first order solver
                            # x = x - 1 / (2*rhos[t_i]) * (x - y_t) * mask 
                            pass

                    if (model_out_type == 'pred_xstart') and not (seq[i] == seq[-1]):
                        # x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x) # x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                        
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                        # calculate \hat{\eposilon}
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                        x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                    + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                        
                    # set back to x_t from x_{t-1}
                    if u < iter_num_U-1 and seq[i] != seq[-1]:
                        x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)

                # save the process
                x_0 = (x/2+0.5)
                if save_progressive and (seq[i] in progress_seq):
                    x_show = x_0.clone().detach().cpu().numpy()       #[0,1]
                    x_show = np.squeeze(x_show)
                    if x_show.ndim == 3:
                        x_show = np.transpose(x_show, (1, 2, 0))
                    progress_img.append(x_show)
                    if log_process:
                        logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                    if show_img:
                        util.imshow(x_show)

            # recover conditional part
            if generate_mode in ['repaint','DDPIR']:
                x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

            # --------------------------------
            # (4) save process
            # --------------------------------
            
            img_E = util.tensor2uint(x)
                    
            if calc_LPIPS:
                img_H_tensor = np.transpose(img_H, (2, 0, 1))
                img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                img_H_tensor = img_H_tensor / 255 * 2 -1
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                test_results['lpips'].append(lpips_score.cpu().detach().numpy()[0][0][0][0])

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

            if save_L:
                util.imsave(img_L, os.path.join(E_path, img_name+'_L.png'))

            if save_LEH:
                util.imsave(np.concatenate([img_L, img_E, img_H], axis=1), os.path.join(E_path, img_name+model_name+'_LEH.png'))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                if generate_mode in ['repaint','DDPIR']:
                    mask = np.squeeze(mask.cpu().numpy())
                    if mask.ndim == 3:
                        mask = np.transpose(mask, (1, 2, 0))
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total,figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+'_process_lambda_{:.3f}_{}.png'.format(lambda_,current_time)))
                images = []
                y_t = np.squeeze((y/2+0.5).cpu().numpy())
                if y_t.ndim == 3:
                    y_t = np.transpose(y_t, (1, 2, 0))
                if generate_mode in ['repaint','DDPIR']:
                    for x in progress_img:
                        images.append((y_t)* mask+ (1-mask) * x)
                    img_total = cv2.hconcat(images)
                    if show_img:
                        util.imshow(img_total,figsize=(80,4))
                    if save_progressive_mask:
                        util.imsave(img_total*255., os.path.join(E_path, img_name+'_process_mask_lambda_{:.3f}_{}.png'.format(lambda_,current_time)))
            logger.info('inpainting complete!')

            # test with the first image in the path
            break

        # --------------------------------
        # Average LPIPS
        # --------------------------------

        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}), sigma: ({:.2f}): {:.2f}'.format(testset_name, noise_level_model, ave_lpips))



    # experiments
    lambdas = [lambda_*i for i in range(1,2)]
    for lambda_ in lambdas:
        test_rho(lambda_)

if __name__ == '__main__':

    main()
