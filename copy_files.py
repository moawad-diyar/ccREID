#!/usr/bin/env python3
"""
Script to copy files from Simple-CCReID to simple-ccReID-refactored
"""
import os
import shutil
from pathlib import Path

# Define source and destination
src_base = Path("../Simple-CCReID")
dst_base = Path(".")

# Files that can be copied as-is (no changes needed)
files_to_copy = [
    # Data files
    "data/img_transforms.py",
    "data/spatial_transforms.py",
    "data/temporal_transforms.py",
    # Dataset files
    "data/datasets/ltcc.py",
    "data/datasets/prcc.py",
    "data/datasets/last.py",
    "data/datasets/ccvid.py",
    "data/datasets/deepchange.py",
    "data/datasets/vcclothes.py",
    # Model files
    "models/classifier.py",
    "models/img_resnet.py",
    "models/vid_resnet.py",
    # Model utils
    "models/utils/c3d_blocks.py",
    "models/utils/inflate.py",
    "models/utils/nonlocal_blocks.py",
    "models/utils/pooling.py",
    # Tools
    "tools/eval_metrics.py",
    # Loss files (most can be copied as-is)
    "losses/arcface_loss.py",
    "losses/circle_loss.py",
    "losses/contrastive_loss.py",
    "losses/cosface_loss.py",
    "losses/cross_entropy_loss_with_label_smooth.py",
    "losses/triplet_loss.py",
    # Other
    "LICENSE",
]

def main():
    for file_path in files_to_copy:
        src_file = src_base / file_path
        dst_file = dst_base / file_path
        
        # Create destination directory if it doesn't exist
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {file_path}")
        else:
            print(f"Warning: Source file not found: {file_path}")

if __name__ == "__main__":
    main()
    print("\nFile copying complete!")
