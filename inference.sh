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
cd seed-vc

# python inference.py \
#     --cache_dir $HF_HOME \
#     --checkpoint /home/shw002/u/seed-vc/models/DiT_uvit_tat_xlsr_ema.pth \
#     --config /home/shw002/u/seed-vc/models/config_dit_mel_seed_uvit_xlsr_tiny.yml \
#     --target /home/shw002/sgile/data/LJSpeech-1.1/wavs/LJ045-0240.wav \
#     singular \
#         --source ../StyleTTS2/out/average/4a45e4b0-b95f-4590-a0ed-9949f06fdfbb/3f6c935bce2432fffbbcddfbf14f0e33f25097d14ff37046ef076fea7e15885d.wav

python inference.py \
    --cache_dir $HF_HOME \
    --checkpoint /home/shw002/u/seed-vc/models/DiT_uvit_tat_xlsr_ema.pth \
    --config /home/shw002/u/seed-vc/models/config_dit_mel_seed_uvit_xlsr_tiny.yml \
    --target /home/shw002/sgile/data/LJSpeech-1.1/wavs/LJ045-0240.wav \
    directory \
        --dir ../StyleTTS2/out/average