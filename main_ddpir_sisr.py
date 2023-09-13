import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_resizer import Resizer
from functools import partial

# from guided_diffusion import dist_util
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

    noise_level_img         = 12.75/255.0       # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusino model
    testset_name            = 'demo_test'    # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps     = 1000
    iter_num                = 100                # set number of sampling iterations
    iter_num_U              = 1                 # set number of inner iterations, default: 1
    skip                    = num_train_timesteps//iter_num     # skip interval
    sr_mode                 = 'blur'            # 'blur', 'cubic' mode of sr up/down sampling

    show_img                = False             # default: False
    save_L                  = True              # save LR image
    save_E                  = False             # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = True              # save generation process

    sigma                   = max(0.001,noise_level_img)  # noise level associated with condition y
    lambda_                 = 1.                # key parameter lambda
    sub_1_analytic          = True              # use analytical solution

    log_process             = False
    ddim_sample             = False             # sampling method
    model_output_type       = 'pred_xstart'     # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode           = 'DiffPIR'         # DiffPIR; DPS; vanilla
    skip_type               = 'quad'            # uniform, quad
    eta                     = 0.                # eta for ddim sampling
    zeta                    = 0.1               
    guidance_scale          = 1.0   

    test_sf                 = [4]               # set scale factor, default: [2, 3, 4], [2], [3], [4]
    inIter                  = 1                 # iter num for sr solution: 4-6
    gamma                   = 1/100             # coef for iterative sr solver 20steps: 0.05-0.10 for zeta=1, 0.09-0.13 for zeta=0 
    classical_degradation   = False             # set classical degradation or bicubic degradation
    task_current            = 'sr'              # 'sr' for super resolution
    n_channels              = 3                 # fixed
    cwd                     = '' 
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results')      # fixed
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{sr_mode}{str(test_sf)}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    calc_LPIPS              = True

    # noise schedule 
    beta_start              = 0.1 / 1000
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
    t_start                 = num_train_timesteps - 1   

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
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, sr_mode:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, sr_mode, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('analytic iter num:{}, gamma:{}'.format(inIter, gamma))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    # --------------------------------
    # load kernel
    # --------------------------------

    # kernels = hdf5storage.loadmat(os.path.join('kernels', 'Levin09.mat'))['kernels']
    if classical_degradation:
        kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'kernels_12.mat'))['kernels']
    else:
        kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'kernels_bicubicx234.mat'))['kernels']

    test_results_ave = OrderedDict()
    test_results_ave['psnr_sf_k'] = []
    test_results_ave['psnr_y_sf_k'] = []
    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        test_results_ave['lpips'] = []

    for sf in test_sf:
        border = sf
        k_num = 8 if classical_degradation else 1

        for k_index in range(k_num):
            logger.info('--------- sf:{:>1d} --k:{:>2d} ---------'.format(sf, k_index))

            if not classical_degradation:  # for bicubic degradation
                k_index = sf-2 if sf < 5 else 2
            k = kernels[0, k_index].astype(np.float64)

            util.surf(k) if show_img else None

            def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type): 
                logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, inIter:{:.3f}, gamma:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, inIter, gamma, guidance_scale))
                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['psnr_y'] = []
                if calc_LPIPS:
                    test_results['lpips'] = []
                for idx, img in enumerate(L_paths):
                    model_out_type = model_output_type

                    # --------------------------------
                    # (1) get img_L
                    # --------------------------------

                    img_name, ext = os.path.splitext(os.path.basename(img))
                    img_H = util.imread_uint(img, n_channels=n_channels)
                    img_H = util.modcrop(img_H, sf)  # modcrop

                    if sr_mode == 'blur':
                        if classical_degradation:
                            img_L = sr.classical_degradation(img_H, k, sf)
                            util.imshow(img_L) if show_img else None
                            img_L = util.uint2single(img_L)
                        else:
                            img_L = util.imresize_np(util.uint2single(img_H), 1/sf)
                    elif sr_mode == 'cubic':
                        img_H_tensor = np.transpose(img_H, (2, 0, 1))
                        img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                        img_H_tensor = img_H_tensor / 255
                        # set up resizers
                        up_sample = partial(F.interpolate, scale_factor=sf)
                        down_sample = Resizer(img_H_tensor.shape, 1/sf).to(device)
                        img_L = down_sample(img_H_tensor)
                        img_L = img_L.cpu().numpy()       #[0,1]
                        img_L = np.squeeze(img_L)
                        if img_L.ndim == 3:
                            img_L = np.transpose(img_L, (1, 2, 0))

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
                        sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
                        if model_out_type == 'pred_xstart' and generate_mode == 'DiffPIR':
                            sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                        #elif model_out_type == 'pred_x_prev':
                        else:
                            sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                        rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))
                            
                    rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)
                    
                    # --------------------------------
                    # (3) initialize x, and pre-calculation
                    # --------------------------------

                    x = cv2.resize(img_L, (img_L.shape[1]*sf, img_L.shape[0]*sf), interpolation=cv2.INTER_CUBIC)
                    if np.ndim(x)==2:
                        x = x[..., None]

                    if classical_degradation:
                        x = sr.shift_pixel(x, sf)
                    x = util.single2tensor4(x).to(device)

                    y = util.single2tensor4(img_L).to(device)   #(1,3,256,256) [0,1]

                    # x = torch.randn_like(x)
                    x = sqrt_alphas_cumprod[t_start] * (2*x-1) + sqrt_1m_alphas_cumprod[t_start] * torch.randn_like(x)

                    k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device) 

                    FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, sf)

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
                    progress_seq = seq[::max(len(seq)//10,1)]
                    if progress_seq[-1] != seq[-1]:
                        progress_seq.append(seq[-1])
                    
                    # reverse diffusion for one image from random noise
                    for i in range(len(seq)):
                        curr_sigma = sigmas[seq[i]].cpu().numpy()
                        # time step associated with the noise level sigmas[i]
                        t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
                        # skip iters
                        if t_i > t_start:
                            continue
                        # repeat for semantic consistence: from repaint
                        for u in range(iter_num_U):
                            # --------------------------------
                            # step 1, reverse diffsuion step
                            # --------------------------------

                            ### solve equation 6b with one reverse diffusion step
                            if 'DPS' in generate_mode:
                                x = x.requires_grad_()
                                xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start', \
                                            model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                            else:
                                x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                        model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                            # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                            #       model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                            # --------------------------------
                            # step 2, FFT
                            # --------------------------------

                            if seq[i] != seq[-1]:
                                if generate_mode == 'DiffPIR':
                                    if sub_1_analytic:
                                        if model_out_type == 'pred_xstart':
                                            # when noise level less than given image noise, skip
                                            if i < num_train_timesteps-noise_model_t: 
                                                if sr_mode == 'blur':
                                                    tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                                    x0_p = x0 / 2 + 0.5
                                                    x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, sf)
                                                    x0_p = x0_p * 2 - 1
                                                    # effective x0
                                                    x0 = x0 + guidance_scale * (x0_p-x0)
                                                elif sr_mode == 'cubic': 
                                                    # iterative back-projection (IBP) solution
                                                    for _ in range(inIter):
                                                        x0 = x0 / 2 + 0.5
                                                        x0 = x0 + gamma * up_sample((y - down_sample(x0))) / (1+rhos[t_i])
                                                        x0 = x0 * 2 - 1
                                            else:
                                                model_out_type = 'pred_x_prev'
                                                x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,model_out_type=model_out_type, \
                                                        model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                                                # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                                                #       model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                                                pass
                                    else:
                                        # zeta=0.25; lambda_=15: FFHQ
                                        # zeta=0.35; lambda_=35: ImageNet
                                        x0 = x0.requires_grad_()
                                        # first order solver
                                        down_sample = Resizer(x.shape, 1/sf).to(device)
                                        #norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x0/2+0.5, x_hat=x0, measurement=y)
                                        norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x0, x_hat=x0, measurement=2*y-1)
                                                            
                                        x0 = x0 - norm_grad * norm / (rhos[t_i]) 
                                        x0 = x0.detach_()
                                        pass                          
                                elif 'DPS' in generate_mode:
                                    down_sample = Resizer(x.shape, 1/sf).to(device)                        
                                    if generate_mode == 'DPS_y0':
                                        norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x, x_hat=x0, measurement=2*y-1)
                                        #norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=xt, x_hat=x0, measurement=2*y-1)    # does not work
                                        x = xt - norm_grad * 1. #norm / (2*rhos[t_i]) 
                                        x = x.detach_()
                                        pass
                                    elif generate_mode == 'DPS_yt':
                                        y_t = sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(y) # add AWGN
                                        #y_t = y_t/2 + 0.5
                                        #norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x, x_hat=xt, measurement=y_t)    # no need to use
                                        norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=xt, x_hat=xt, measurement=y_t)
                                        x = xt - norm_grad * lambda_ * norm / (rhos[t_i]) * 0.35
                                        x = x.detach_()
                                        pass
                                
                            # add noise back to t=i-1
                            if (generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == iter_num_U-1):
                                #x = sqrt_alphas_cumprod[t_i] * (x0) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                                
                                t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                                eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                                # calculate \hat{\eposilon}
                                eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                                x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                            + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                            else:
                                #x = x0
                                pass
                                
                            # set back to x_t from x_{t-1}
                            if u < iter_num_U-1 and seq[i] != seq[-1]:
                                ### it's equivalent to use x & xt (?), but with xt the computation is faster.
                                # x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)
                                sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                                x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                                        sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)
                                    

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

                    # --------------------------------
                    # (3) img_E
                    # --------------------------------

                    img_E = util.tensor2uint(x_0)

                    psnr = util.calculate_psnr(img_E, img_H, border=border)
                    test_results['psnr'].append(psnr)
                    
                    if calc_LPIPS:
                        img_H_tensor = np.transpose(img_H, (2, 0, 1))
                        img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                        img_H_tensor = img_H_tensor / 255 * 2 -1
                        lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                        lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                        test_results['lpips'].append(lpips_score)
                        logger.info('{:->4d}--> {:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(idx+1, img_name+ext, sf, k_index, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
                    else:
                        logger.info('{:->4d}--> {:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.4f}dB'.format(idx+1, img_name+ext, sf, k_index, psnr))

                    if save_E:
                        util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_'+model_name+ext))

                    if n_channels == 1:
                        img_H = img_H.squeeze()

                    if save_progressive:
                        now = datetime.now()
                        current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                        img_total = cv2.hconcat(progress_img)
                        if show_img:
                            util.imshow(img_total,figsize=(80,4))
                        util.imsave(img_total*255., os.path.join(E_path, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}{}'.format(noise_level_img,lambda_,current_time,psnr,ext)))
                        
                    # --------------------------------
                    # (4) img_LEH
                    # --------------------------------

                    img_L = util.single2uint(img_L).squeeze()

                    if save_LEH:
                        k_v = k/np.max(k)*1.0
                        if n_channels==1:
                            k_v = util.single2uint(k_v)
                        else:
                            k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, n_channels]))
                        k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                        img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                        img_I[:k_v.shape[0], -k_v.shape[1]:, ...] = k_v
                        img_I[:img_L.shape[0], :img_L.shape[1], ...] = img_L
                        util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if show_img else None
                        util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_LEH'+ext))

                    if save_L:
                        util.imsave(img_L, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_LR'+ext))

                    if n_channels == 3:
                        img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                        img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                        psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                        test_results['psnr_y'].append(psnr_y)
                    
                # --------------------------------
                # Average PSNR and LPIPS for all images
                # --------------------------------

                ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
                logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f} dB'.format(testset_name, sf, k_index, noise_level_model, ave_psnr_k))
                test_results_ave['psnr_sf_k'].append(ave_psnr_k)

                if n_channels == 3:  # RGB image
                    ave_psnr_y_k = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    logger.info('------> Average PSNR(Y) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f} dB'.format(testset_name, sf, k_index, noise_level_model, ave_psnr_y_k))
                    test_results_ave['psnr_y_sf_k'].append(ave_psnr_y_k)

                if calc_LPIPS:
                    ave_lpips_k = sum(test_results['lpips']) / len(test_results['lpips'])
                    logger.info('------> Average LPIPS of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f}'.format(testset_name, sf, k_index, noise_level_model, ave_lpips_k))
                    test_results_ave['lpips'].append(ave_lpips_k)    
                return test_results_ave

            # experiments
            lambdas = [lambda_*i for i in range(2,13)]
            for lambda_ in lambdas:
                #for zeta_i in [zeta*i for i in range(2,4)]:
                for zeta_i in [0.25]:
                    test_results_ave = test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type)

    # ---------------------------------------
    # Average PSNR and LPIPS for all sf and kernels
    # ---------------------------------------

    ave_psnr_sf_k = sum(test_results_ave['psnr_sf_k']) / len(test_results_ave['psnr_sf_k'])
    logger.info('------> Average PSNR of ({}) {:.4f} dB'.format(testset_name, ave_psnr_sf_k))
    if n_channels == 3:
        ave_psnr_y_sf_k = sum(test_results_ave['psnr_y_sf_k']) / len(test_results_ave['psnr_y_sf_k'])
        logger.info('------> Average PSNR-Y of ({}) {:.4f} dB'.format(testset_name, ave_psnr_y_sf_k))
    if calc_LPIPS:
        ave_lpips_sf_k = sum(test_results_ave['lpips']) / len(test_results_ave['lpips'])
        logger.info('------> Average LPIPS of ({}) {:.4f}'.format(testset_name, ave_lpips_sf_k))

if __name__ == '__main__':

    main()
