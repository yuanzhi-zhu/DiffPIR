#!/bin/bash

URL_1="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
OUTPUT_FILE_1="model_zoo/256x256_diffusion_uncond.pt"
wget $URL_1 -O $OUTPUT_FILE_1

pip install gdown==4.7.1
FILE_ID="1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh"
OUTPUT_FILE_2="model_zoo/diffusion_ffhq_10m.pt"
gdown --id $FILE_ID -O $OUTPUT_FILE_2
