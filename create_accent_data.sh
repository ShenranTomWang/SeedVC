#!/bin/bash

#SBATCH --job-name=seed_vc-create_accent_data
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
export ROOT=/home/shw002/u/data/LibriTTS-train

for target in \
    /home/aip000/u/sgile/crk_corpus_24k/wavs_24k/wavs/cdd6388e-9813-448b-b025-6bf759af3f0f/42505a4ddccd74adec7835c2628dbe2b416ef9ad949ed98494202c29fafb9c62.wav \
    /home/aip000/u/sgile/crk_corpus_24k/wavs_24k/wavs/79e8aacd-539a-4870-b731-89fc2d5f239b/9a1597a0de3b48bcb404b66f41620e94f5a24b6f7cc4178b300be90257c76d79.wav \
    /home/aip000/u/sgile/crk_corpus_24k/wavs_24k/wavs/131fd554-b01b-4153-8143-a73784cf7df8/962b1eefba4c11152ebef0e3be0741dc1fa055ab4a8e47f80353e2b4f1f15807.wav \
    /home/aip000/u/sgile/crk_corpus_24k/wavs_24k/wavs/d5210fe2-01c8-4af3-9686-8c3f3b24f139/7d63c313ec3410a2933871e98cd1442c0d6118478848bd2aabc215fcbfc446a1.wav \
    /home/aip000/u/sgile/crk_corpus_24k/wavs_24k/wavs/290037ac-9ad3-491d-9517-5c3c4cef89ac/968ac4b7308b87654656691c46c582b3b348edda41ff57cb758e6dfd693c26f3.wav \
    /home/aip000/u/sgile/crk_corpus_24k/wavs_24k/wavs/63e6f575-47af-4396-a765-ce390292593d/90eebb2a412e74df0777d72842662553c9b19e5fc410c7ef78dd53efb8b1bdea.wav \
    /home/aip000/u/sgile/crk_corpus_24k/wavs_24k/wavs/912b397e-4a19-4ee7-b127-ba367db2b6da/9e84b636802cae1da22525a3419361d5d4938955db50d119d8879a4d7e4ac5fd.wav
do
    IFS="/" read -ra wav_name_split <<< "$wav_name"
    python inference_v2.py \
        --ar-checkpoint-path "/home/shw002/u/SeedVC/models/v2/ar_base.pth" \
        --cfm-checkpoint-path "/home/shw002/u/SeedVC/models/v2/cfm_small.pth" \
        --target $target \
        --output /home/shw002/u/data/LibriTTS-train-${wav_name_split[7]} \
        --anonymization-only false \
        --convert-style true \
        --similarity-cfg-rate 1.0 \
        directory \
            --dir $ROOT \
            --root $ROOT
done