#!/usr/bin/env python3
"""
Setup script to copy remaining files from Simple-CCReID to simple-ccReID-refactored
Run this from the WaitingZone directory: python simple-ccReID-refactored/setup_copy_files.py
"""
import os
import shutil
from pathlib import Path

def main():
    # Define paths
    src_base = Path("Simple-CCReID")
    dst_base = Path("simple-ccReID-refactored")
    
    if not src_base.exists():
        print(f"Error: Source directory '{src_base}' not found!")
        print("Please run this script from the WaitingZone directory.")
        return
    
    if not dst_base.exists():
        print(f"Error: Destination directory '{dst_base}' not found!")
        return
    
    # Files to copy (these don't need modifications)
    files_to_copy = [
        # Data transform files
        ("data/img_transforms.py", "data/img_transforms.py"),
        ("data/spatial_transforms.py", "data/spatial_transforms.py"),
        ("data/temporal_transforms.py", "data/temporal_transforms.py"),
        
        # Dataset loaders
        ("data/datasets/ltcc.py", "data/datasets/ltcc.py"),
        ("data/datasets/prcc.py", "data/datasets/prcc.py"),
        ("data/datasets/last.py", "data/datasets/last.py"),
        ("data/datasets/ccvid.py", "data/datasets/ccvid.py"),
        ("data/datasets/deepchange.py", "data/datasets/deepchange.py"),
        ("data/datasets/vcclothes.py", "data/datasets/vcclothes.py"),
        
        # Model files
        ("models/classifier.py", "models/classifier.py"),
        ("models/img_resnet.py", "models/img_resnet.py"),
        ("models/vid_resnet.py", "models/vid_resnet.py"),
        
        # Model utils
        ("models/utils/c3d_blocks.py", "models/utils/c3d_blocks.py"),
        ("models/utils/inflate.py", "models/utils/inflate.py"),
        ("models/utils/nonlocal_blocks.py", "models/utils/nonlocal_blocks.py"),
        ("models/utils/pooling.py", "models/utils/pooling.py"),
        
        # Loss functions
        ("losses/arcface_loss.py", "losses/arcface_loss.py"),
        ("losses/circle_loss.py", "losses/circle_loss.py"),
        ("losses/contrastive_loss.py", "losses/contrastive_loss.py"),
        ("losses/cosface_loss.py", "losses/cosface_loss.py"),
        ("losses/cross_entropy_loss_with_label_smooth.py", "losses/cross_entropy_loss_with_label_smooth.py"),
        ("losses/triplet_loss.py", "losses/triplet_loss.py"),
        
        # Tools
        ("tools/eval_metrics.py", "tools/eval_metrics.py"),
        
        # License
        ("LICENSE", "LICENSE"),
    ]
    
    print("Starting file copy process...")
    print("=" * 60)
    
    copied_count = 0
    failed_count = 0
    
    for src_rel, dst_rel in files_to_copy:
        src_file = src_base / src_rel
        dst_file = dst_base / dst_rel
        
        # Create destination directory if it doesn't exist
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        if src_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
                print(f"✓ Copied: {src_rel}")
                copied_count += 1
            except Exception as e:
                print(f"✗ Failed to copy {src_rel}: {e}")
                failed_count += 1
        else:
            print(f"✗ Source file not found: {src_rel}")
            failed_count += 1
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Successfully copied: {copied_count} files")
    print(f"  Failed: {failed_count} files")
    
    if failed_count == 0:
        print("\n✓ Setup complete! All files copied successfully.")
        print("\nNext steps:")
        print("  1. cd simple-ccReID-refactored")
        print("  2. pip install -r requirements.txt")
        print("  3. python main.py --help  # Test the installation")
        print("\nSee README.md for training instructions.")
    else:
        print("\n⚠ Some files failed to copy. Please check the errors above.")
        print("You may need to copy these files manually.")

if __name__ == "__main__":
    main()
