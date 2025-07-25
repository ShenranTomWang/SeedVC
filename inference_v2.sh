#!/bin/bash

#SBATCH --job-name=seed_vc-inference_v2
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
export HF_HOME=/home/shw002/u/tmp
export ROOT=/home/shw002/StyleTTS2/out

for wav_name in \
    27_123349_000001_000000 \
    40_121026_000008_000000 \
    118_47824_000003_000001
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    echo "Processing target: $target"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output out/v2/$wav_name \
        --anonymization-only false \
        --convert-style true \
        --similarity-cfg-rate 1.0 \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done