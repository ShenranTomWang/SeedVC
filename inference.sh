python inference.py
    --cache_dir $HF_HOME
    singular
        --source ../StyleTTS2/out/average/4a45e4b0-b95f-4590-a0ed-9949f06fdfbb/3f6c935bce2432fffbbcddfbf14f0e33f25097d14ff37046ef076fea7e15885d.wav
        --output out
        --checkpoint /home/shw002/u/seed-vc/models/DiT_uvit_tat_xlsr_ema.pth
        --config /home/shw002/u/seed-vc/models/config_dit_mel_seed_uvit_xlsr_tiny.yml

python inference.py
    --cache_dir $HF_HOME
    directory
        --dir ../StyleTTS2/out/average
        --output out
        --checkpoint /home/shw002/u/seed-vc/models/DiT_uvit_tat_xlsr_ema.pth
        --config /home/shw002/u/seed-vc/models/config_dit_mel_seed_uvit_xlsr_tiny.yml