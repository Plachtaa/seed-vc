#!/bin/bash

# 设置环境变量
export HF_HUB_CACHE='./checkpoints/hf_cache'
export HF_ENDPOINT='https://hf-mirror.com'

# 运行训练脚本
python train.py \
    --config configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml \
    --pretrained-ckpt modeldata/seed_vc/DiT_uvit_tat_xlsr_ema.pth \
    --dataset-dir dataset/八重神子/ \
    --run-name real_time_bachong \
    --batch-size 4 \
    --max-steps 1000 \
    --max-epochs 100 \
    --save-every 500 \
    --num-workers 5 \
    --gpu 0
