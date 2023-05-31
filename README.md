# [Denoising Diffusion Models for Plug-and-Play Image Restoration](https://arxiv.org/abs/2305.08995)

[Yuanzhi Zhu](https://yuanzhi-zhu.github.io/about/), [Kai Zhang](https://cszn.github.io/), [Jingyun Liang](https://jingyunliang.github.io/), [Jiezhang Cao](https://www.jiezhangcao.com/), [Bihan Wen](https://personal.ntu.edu.sg/bihan.wen/), Radu Timofte, Luc Van Gool.

This repository contains the code and data associated with the paper "Denoising Diffusion Models for Plug-and-Play Image Restoration", which was presented at the CVPR workshop NTIRE 2023.

This code is based on the [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion) and [DPIR](https://github.com/cszn/DPIR).

## Abstract

Plug-and-play Image Restoration (IR) has been widely recognized as a flexible and interpretable method for solving various inverse problems by utilizing any off-the-shelf denoiser as the implicit image prior. However, most existing methods focus on discriminative Gaussian denoisers. Although diffusion models have shown impressive performance for high-quality image synthesis, their potential to serve as a generative denoiser prior to the plug-and-play IR methods remains to be further explored.
While several other attempts have been made to adopt diffusion models for image restoration, they either fail to achieve satisfactory results or typically require an unacceptable number of Neural Function Evaluations (NFEs) during inference.
This paper proposes DiffPIR, which integrates the traditional plug-and-play method into the diffusion sampling framework. Compared to plug-and-play IR methods that rely on discriminative Gaussian denoisers, DiffPIR is expected to inherit the generative ability of diffusion models. Experimental results on three representative IR tasks, including super-resolution, image deblurring, and inpainting, demonstrate that DiffPIR achieves state-of-the-art performance on both the FFHQ and ImageNet datasets in terms of reconstruction faithfulness and perceptual quality with no more than 100 NFEs.

## Clone and Install
```
git clone https://github.com/yuanzhi-zhu/DiffPIR.git
cd DiffPIR
pip install -r requirements.txt
```

for motion blur, also download https://github.com/LeviBorodenko/motionblur to the DiffPIR folder.

links to model checkpoints can be found in ./model_zoo/README.md

## Citation

If you find this repo helpful, please cite:

```

@misc{zhu2023denoising,
      title={Denoising Diffusion Models for Plug-and-Play Image Restoration}, 
      author={Yuanzhi Zhu and Kai Zhang and Jingyun Liang and Jiezhang Cao and Bihan Wen and Radu Timofte and Luc Van Gool},
      year={2023},
      eprint={2305.08995},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```