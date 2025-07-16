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
export HF_HOME=/home/shw002/u/tmp
export ROOT=/home/shw002/StyleTTS2/out
export CHECKPOINT_NAME=DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned
export CONFIG_NAME=config_dit_mel_seed_uvit_whisper_small_wavenet

for wav_name in \
    27_123349_000001_000000 \
    40_121026_000008_000000 \
    118_47824_000003_000001
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name_split[2]}_${wav_name_split[3]}.wav"
    for parent_dir in \
        baseline \
        average \
        gender_average \
        gender_average_swapped \
        noise \
        steer_f2m \
        steer_m2f \
        custom_libritts_female \
        custom_libritts_male \
        s_a_average \
        s_a_gender_average \
        s_a_gender_average_swapped \
        s_a_noise \
        s_a_steer_m2f \
        s_a_steer_f2m \
        s_a_custom_libritts_female \
        s_a_custom_libritts_male \
        pca/top_32_average \
        pca/top_32_noise \
        pca/top_32_steer_f2m \
        pca/top_32_steer_m2f \
        pca/top_32_gender_average \
        pca/top_32_gender_average_swapped \
        pca/top_32_custom_libritts_female \
        pca/top_32_custom_libritts_male \
        gender_logistic_regression/top_32_average \
        gender_logistic_regression/top_32_noise \
        gender_logistic_regression/top_32_steer_f2m \
        gender_logistic_regression/top_32_steer_m2f \
        gender_logistic_regression/top_32_gender_average \
        gender_logistic_regression/top_32_gender_average_swapped \
        gender_logistic_regression/top_32_custom_libritts_female \
        gender_logistic_regression/top_32_custom_libritts_male \
        pca/s_a_top_32_average \
        pca/s_a_top_32_noise \
        pca/s_a_top_32_steer_f2m \
        pca/s_a_top_32_steer_m2f \
        pca/s_a_top_32_gender_average \
        pca/s_a_top_32_gender_average_swapped \
        pca/s_a_top_32_custom_libritts_female \
        pca/s_a_top_32_custom_libritts_male \
        gender_logistic_regression/s_a_top_32_average \
        gender_logistic_regression/s_a_top_32_noise \
        gender_logistic_regression/s_a_top_32_steer_f2m \
        gender_logistic_regression/s_a_top_32_steer_m2f \
        gender_logistic_regression/s_a_top_32_gender_average \
        gender_logistic_regression/s_a_top_32_gender_average_swapped \
        gender_logistic_regression/s_a_top_32_custom_libritts_female \
        gender_logistic_regression/s_a_top_32_custom_libritts_male
    do
        python inference.py \
            --checkpoint /home/shw002/u/SeedVC/models/$CHECKPOINT_NAME.pth \
            --config configs/presets/$CONFIG_NAME.yml \
            --target $target \
            --output out/v1/$CHECKPOINT_NAME/$wav_name \
            directory \
                --dir $ROOT/$parent_dir \
                --root $ROOT
    done
done