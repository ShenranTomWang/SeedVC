#!/bin/bash

#SBATCH --job-name=inference_seed-vc
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=10080
#SBATCH --partition=gpu_a100
#SBATCH --account=nrc_ict__gpu_a100
#SBATCH --gres=gpu:1

cd /home/shw002
source .bashrc
conda activate VC
cd SeedVC

python inference.py \
    --cache_dir $HF_HUB_CACHE \
    --checkpoint /home/shw002/u/SeedVC/models/DiT_uvit_tat_xlsr_ema.pth \
    --config /home/shw002/u/SeedVC/models/config_dit_mel_seed_uvit_xlsr_tiny.yml \
    --target /home/shw002/u/data/LibriTTS_R/train-clean-100/27/123349/27_123349_000001_000000.wav \
    directory \
        --dir ../StyleTTS2/out/average