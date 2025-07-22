#!/bin/bash

#SBATCH --job-name=seed_vc-inference_xlsr
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
export CHECKPOINT_NAME=DiT_uvit_tat_xlsr_ema
export CONFIG_NAME=config_dit_mel_seed_uvit_xlsr_tiny

for wav_name in \
    27_123349_000001_000000 \
    40_121026_000008_000000 \
    118_47824_000003_000001
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    python inference.py \
        --checkpoint /home/shw002/u/SeedVC/models/$CHECKPOINT_NAME.pth \
        --config configs/presets/$CONFIG_NAME.yml \
        --target $target \
        --output out/v1/$CHECKPOINT_NAME/$wav_name \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done