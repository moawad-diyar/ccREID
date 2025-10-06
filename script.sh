#!/bin/bash
# Training scripts for Simple-CCReID Refactored (Single GPU)
# Updated for single GPU training without distributed setup

# For LTCC dataset
python main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0

# For PRCC dataset
python main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0

# For VC-Clothes dataset
python main.py --dataset vcclothes --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0

# For VC-Clothes evaluation (clothes-changing)
python main.py --dataset vcclothes_cc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0 --eval --resume ./logs/vcclothes/res50-cels-cal/best_model.pth.tar

# For VC-Clothes evaluation (same clothes)
python main.py --dataset vcclothes_sc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0 --eval --resume ./logs/vcclothes/res50-cels-cal/best_model.pth.tar

# For DeepChange dataset (with mixed precision for speed)
python main.py --dataset deepchange --cfg configs/res50_cels_cal_16x4.yaml --root /path/to/data --output ./logs --gpu 0 --amp

# For LaST dataset (with mixed precision for speed)
python main.py --dataset last --cfg configs/res50_cels_cal_tri_16x4.yaml --root /path/to/data --output ./logs --gpu 0 --amp

# For CCVID dataset (video-based)
python main.py --dataset ccvid --cfg configs/c2dres50_ce_cal.yaml --root /path/to/data --output ./logs --gpu 0
