#!/bin/bash

URL_1="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
OUTPUT_FILE_1="model_zoo/256x256_diffusion_uncond.pt"
wget $URL_1 -O $OUTPUT_FILE_1

pip install gdown
FILE_ID="1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh"
OUTPUT_FILE_2="model_zoo/diffusion_ffhq_10m.pt"
gdown --id $FILE_ID -O $OUTPUT_FILE_2
