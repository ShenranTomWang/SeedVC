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
export ROOT=/home/shw002/u/StyleTTS2/out/crk

for wav_name in \
    27_123349_000001_000000 \
    201_122255_000001_000000 \
    118_47824_000003_000001 \
    1081_128618_000005_000000 \
    2002_139469_000002_000000 \
    19_198_000000_000000 \
    40_121026_000008_000000 \
    125_121124_000009_000003 \
    103_1241_000004_000002 \
    298_126790_000008_000000
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    echo "Processing target: $target"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output out/v2/similarity=1/$wav_name \
        --anonymization-only false \
        --convert-style true \
        --similarity-cfg-rate 1.0 \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done

for wav_name in \
    27_123349_000001_000000 \
    201_122255_000001_000000 \
    118_47824_000003_000001 \
    1081_128618_000005_000000 \
    2002_139469_000002_000000 \
    19_198_000000_000000 \
    40_121026_000008_000000 \
    125_121124_000009_000003 \
    103_1241_000004_000002 \
    298_126790_000008_000000
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    echo "Processing target: $target"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output out/v2/default/$wav_name \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done

for wav_name in \
    27_123349_000001_000000 \
    201_122255_000001_000000 \
    118_47824_000003_000001 \
    1081_128618_000005_000000 \
    2002_139469_000002_000000 \
    19_198_000000_000000 \
    40_121026_000008_000000 \
    125_121124_000009_000003 \
    103_1241_000004_000002 \
    298_126790_000008_000000
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    echo "Processing target: $target"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output out/v2/similarity=0/$wav_name \
        --anonymization-only false \
        --convert-style true \
        --similarity-cfg-rate 0 \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done

for wav_name in \
    27_123349_000001_000000 \
    201_122255_000001_000000 \
    118_47824_000003_000001 \
    1081_128618_000005_000000 \
    2002_139469_000002_000000 \
    19_198_000000_000000 \
    40_121026_000008_000000 \
    125_121124_000009_000003 \
    103_1241_000004_000002 \
    298_126790_000008_000000
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    echo "Processing target: $target"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output out/v2/similarity=0.3/$wav_name \
        --anonymization-only false \
        --convert-style true \
        --similarity-cfg-rate 0.3 \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done

for wav_name in \
    27_123349_000001_000000 \
    201_122255_000001_000000 \
    118_47824_000003_000001 \
    1081_128618_000005_000000 \
    2002_139469_000002_000000 \
    19_198_000000_000000 \
    40_121026_000008_000000 \
    125_121124_000009_000003 \
    103_1241_000004_000002 \
    298_126790_000008_000000
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    echo "Processing target: $target"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output out/v2/similarity=0.5/$wav_name \
        --anonymization-only false \
        --convert-style true \
        --similarity-cfg-rate 0.5 \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done

for wav_name in \
    27_123349_000001_000000 \
    201_122255_000001_000000 \
    118_47824_000003_000001 \
    1081_128618_000005_000000 \
    2002_139469_000002_000000 \
    19_198_000000_000000 \
    40_121026_000008_000000 \
    125_121124_000009_000003 \
    103_1241_000004_000002 \
    298_126790_000008_000000
do
    IFS="_" read -ra wav_name_split <<< "$wav_name"
    target="/home/shw002/u/data/LibriTTS_R/train-clean-100/${wav_name_split[0]}/${wav_name_split[1]}/${wav_name}.wav"
    echo "Processing target: $target"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output out/v2/convert_style=False/$wav_name \
        --convert-style false \
        directory \
            --dir $ROOT/baseline \
            --root $ROOT
done