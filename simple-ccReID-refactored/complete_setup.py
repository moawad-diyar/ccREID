#!/usr/bin/env python3
"""
Complete setup script to copy all remaining files from Simple-CCReID
Run this from the WaitingZone directory: python simple-ccReID-refactored/complete_setup.py
"""
import os
import shutil
from pathlib import Path

def main():
    print("=" * 70)
    print("Simple-CCReID Refactored - Complete Setup Script")
    print("=" * 70)
    
    # Define paths
    src_base = Path("Simple-CCReID")
    dst_base = Path("simple-ccReID-refactored")
    
    if not src_base.exists():
        print(f"\n❌ Error: Source directory '{src_base}' not found!")
        print("Please run this script from the WaitingZone directory.")
        return False
    
    if not dst_base.exists():
        print(f"\n❌ Error: Destination directory '{dst_base}' not found!")
        return False
    
    # Files to copy
    files_to_copy = [
        # Dataset files
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
        
        # License
        ("LICENSE", "LICENSE"),
    ]
    
    print("\n📋 Files to copy:")
    print(f"   - {len(files_to_copy)} files total")
    print("\n🔄 Starting file copy process...\n")
    
    copied_count = 0
    failed_count = 0
    failed_files = []
    
    for src_rel, dst_rel in files_to_copy:
        src_file = src_base / src_rel
        dst_file = dst_base / dst_rel
        
        # Create destination directory if it doesn't exist
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        if src_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
                print(f"   ✓ {src_rel}")
                copied_count += 1
            except Exception as e:
                print(f"   ✗ {src_rel}: {e}")
                failed_count += 1
                failed_files.append(src_rel)
        else:
            print(f"   ✗ {src_rel} (not found)")
            failed_count += 1
            failed_files.append(src_rel)
    
    print("\n" + "=" * 70)
    print("📊 Summary:")
    print("=" * 70)
    print(f"   ✓ Successfully copied: {copied_count} files")
    print(f"   ✗ Failed: {failed_count} files")
    
    if failed_count == 0:
        print("\n🎉 Setup complete! All files copied successfully.")
        print("\n📝 Next steps:")
        print("   1. cd simple-ccReID-refactored")
        print("   2. pip install -r requirements.txt")
        print("   3. python main.py --help  # Test the installation")
        print("\n📖 See README.md or QUICKSTART.md for training instructions.")
        return True
    else:
        print("\n⚠️  Some files failed to copy:")
        for f in failed_files:
            print(f"      - {f}")
        print("\n💡 You may need to copy these files manually.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
