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

for parent_dir in \
    out/baseline \
    out/average \
    out/gender_average \
    out/gender_average_swapped \
    out/noise \
    out/steer_f2m \
    out/steer_m2f \
    out/custom_libritts_female \
    out/custom_libritts_male \
    out/s_a_average \
    out/s_a_gender_average \
    out/s_a_gender_average_swapped \
    out/s_a_noise \
    out/s_a_steer_m2f \
    out/s_a_steer_f2m \
    out/s_a_custom_libritts_female \
    out/s_a_custom_libritts_male \
    out/pca/top_32_average \
    out/pca/top_32_noise \
    out/pca/top_32_steer_f2m \
    out/pca/top_32_steer_m2f \
    out/pca/top_32_gender_average \
    out/pca/top_32_gender_average_swapped \
    out/pca/top_32_custom_libritts_female \
    out/pca/top_32_custom_libritts_male \
    out/gender_logistic_regression/top_32_average \
    out/gender_logistic_regression/top_32_noise \
    out/gender_logistic_regression/top_32_steer_f2m \
    out/gender_logistic_regression/top_32_steer_m2f \
    out/gender_logistic_regression/top_32_gender_average \
    out/gender_logistic_regression/top_32_gender_average_swapped \
    out/gender_logistic_regression/top_32_custom_libritts_female \
    out/gender_logistic_regression/top_32_custom_libritts_male \
    out/pca/s_a_top_32_average \
    out/pca/s_a_top_32_noise \
    out/pca/s_a_top_32_steer_f2m \
    out/pca/s_a_top_32_steer_m2f \
    out/pca/s_a_top_32_gender_average \
    out/pca/s_a_top_32_gender_average_swapped \
    out/pca/s_a_top_32_custom_libritts_female \
    out/pca/s_a_top_32_custom_libritts_male \
    out/gender_logistic_regression/s_a_top_32_average \
    out/gender_logistic_regression/s_a_top_32_noise \
    out/gender_logistic_regression/s_a_top_32_steer_f2m \
    out/gender_logistic_regression/s_a_top_32_steer_m2f \
    out/gender_logistic_regression/s_a_top_32_gender_average \
    out/gender_logistic_regression/s_a_top_32_gender_average_swapped \
    out/gender_logistic_regression/s_a_top_32_custom_libritts_female \
    out/gender_logistic_regression/s_a_top_32_custom_libritts_male
do
    python inference.py \
        --cache_dir $HF_HUB_CACHE \
        --checkpoint /home/shw002/u/SeedVC/models/DiT_uvit_tat_xlsr_ema.pth \
        --config /home/shw002/u/SeedVC/models/config_dit_mel_seed_uvit_xlsr_tiny.yml \
        --target /home/shw002/u/data/LibriTTS_R/train-clean-100/27/123349/27_123349_000001_000000.wav \
        directory \
            --dir home/shw002/StyleTTS2/$parent_dir
            --root home/shw002/StyleTTS2/out
done