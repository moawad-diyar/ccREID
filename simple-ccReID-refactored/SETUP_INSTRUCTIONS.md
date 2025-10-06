# Setup Instructions for Simple-CCReID Refactored

## Quick Setup Guide

This refactored version has the core training/testing logic updated, but you need to copy some files from the original `Simple-CCReID` directory that don't require changes.

### Step 1: Copy Required Files

Run these commands from the `WaitingZone` directory:

#### On Windows (PowerShell):
```powershell
# Copy dataset loaders
Copy-Item "Simple-CCReID\data\datasets\*.py" "simple-ccReID-refactored\data\datasets\" -Force

# Copy data transform files
Copy-Item "Simple-CCReID\data\img_transforms.py" "simple-ccReID-refactored\data\" -Force
Copy-Item "Simple-CCReID\data\spatial_transforms.py" "simple-ccReID-refactored\data\" -Force
Copy-Item "Simple-CCReID\data\temporal_transforms.py" "simple-ccReID-refactored\data\" -Force

# Copy model files
Copy-Item "Simple-CCReID\models\classifier.py" "simple-ccReID-refactored\models\" -Force
Copy-Item "Simple-CCReID\models\img_resnet.py" "simple-ccReID-refactored\models\" -Force
Copy-Item "Simple-CCReID\models\vid_resnet.py" "simple-ccReID-refactored\models\" -Force

# Copy model utils
New-Item -ItemType Directory -Force -Path "simple-ccReID-refactored\models\utils"
Copy-Item "Simple-CCReID\models\utils\*.py" "simple-ccReID-refactored\models\utils\" -Force

# Copy loss functions
Copy-Item "Simple-CCReID\losses\arcface_loss.py" "simple-ccReID-refactored\losses\" -Force
Copy-Item "Simple-CCReID\losses\circle_loss.py" "simple-ccReID-refactored\losses\" -Force
Copy-Item "Simple-CCReID\losses\contrastive_loss.py" "simple-ccReID-refactored\losses\" -Force
Copy-Item "Simple-CCReID\losses\cosface_loss.py" "simple-ccReID-refactored\losses\" -Force
Copy-Item "Simple-CCReID\losses\cross_entropy_loss_with_label_smooth.py" "simple-ccReID-refactored\losses\" -Force
Copy-Item "Simple-CCReID\losses\triplet_loss.py" "simple-ccReID-refactored\losses\" -Force

# Copy evaluation metrics
Copy-Item "Simple-CCReID\tools\eval_metrics.py" "simple-ccReID-refactored\tools\" -Force

# Copy LICENSE
Copy-Item "Simple-CCReID\LICENSE" "simple-ccReID-refactored\" -Force
```

#### On Linux/Mac:
```bash
# Copy dataset loaders
mkdir -p simple-ccReID-refactored/data/datasets
cp Simple-CCReID/data/datasets/*.py simple-ccReID-refactored/data/datasets/

# Copy data transform files
cp Simple-CCReID/data/img_transforms.py simple-ccReID-refactored/data/
cp Simple-CCReID/data/spatial_transforms.py simple-ccReID-refactored/data/
cp Simple-CCReID/data/temporal_transforms.py simple-ccReID-refactored/data/

# Copy model files
cp Simple-CCReID/models/classifier.py simple-ccReID-refactored/models/
cp Simple-CCReID/models/img_resnet.py simple-ccReID-refactored/models/
cp Simple-CCReID/models/vid_resnet.py simple-ccReID-refactored/models/

# Copy model utils
mkdir -p simple-ccReID-refactored/models/utils
cp Simple-CCReID/models/utils/*.py simple-ccReID-refactored/models/utils/

# Copy loss functions
cp Simple-CCReID/losses/arcface_loss.py simple-ccReID-refactored/losses/
cp Simple-CCReID/losses/circle_loss.py simple-ccReID-refactored/losses/
cp Simple-CCReID/losses/contrastive_loss.py simple-ccReID-refactored/losses/
cp Simple-CCReID/losses/cosface_loss.py simple-ccReID-refactored/losses/
cp Simple-CCReID/losses/cross_entropy_loss_with_label_smooth.py simple-ccReID-refactored/losses/
cp Simple-CCReID/losses/triplet_loss.py simple-ccReID-refactored/losses/

# Copy evaluation metrics
cp Simple-CCReID/tools/eval_metrics.py simple-ccReID-refactored/tools/

# Copy LICENSE
cp Simple-CCReID/LICENSE simple-ccReID-refactored/
```

### Step 2: Install Dependencies

```bash
cd simple-ccReID-refactored
pip install -r requirements.txt
```

### Step 3: Verify Installation

Check that all required files are present:

```bash
# Check data files
ls data/datasets/  # Should show: ccvid.py, deepchange.py, last.py, ltcc.py, prcc.py, vcclothes.py
ls data/           # Should show: __init__.py, dataset_loader.py, img_transforms.py, samplers.py, spatial_transforms.py, temporal_transforms.py

# Check model files
ls models/         # Should show: __init__.py, classifier.py, img_resnet.py, vid_resnet.py, utils/
ls models/utils/   # Should show: c3d_blocks.py, inflate.py, nonlocal_blocks.py, pooling.py

# Check loss files
ls losses/         # Should show: __init__.py, arcface_loss.py, circle_loss.py, clothes_based_adversarial_loss.py, contrastive_loss.py, cosface_loss.py, cross_entropy_loss_with_label_smooth.py, triplet_loss.py

# Check tools
ls tools/          # Should show: eval_metrics.py, utils.py
```

### Step 4: Test Run

Try a quick test to ensure everything is set up correctly:

```bash
python main.py --help
```

You should see the argument parser output without any import errors.

---

## Files Already Refactored

The following files have been updated and are ready to use:

### Core Training Files:
- ✅ `main.py` - Removed apex, distributed training, updated for single GPU
- ✅ `train.py` - Uses PyTorch native AMP instead of apex
- ✅ `test.py` - Removed distributed gathering, simplified for single GPU

### Configuration Files:
- ✅ `configs/default_img.py` - Updated default paths and batch sizes
- ✅ `configs/default_vid.py` - Updated default paths and batch sizes
- ✅ `configs/*.yaml` - Updated batch sizes for single GPU

### Data Loading:
- ✅ `data/__init__.py` - Removed distributed samplers, uses standard DataLoader
- ✅ `data/samplers.py` - Simplified RandomIdentitySampler (removed distributed version)
- ✅ `data/dataset_loader.py` - Standard dataset classes

### Losses:
- ✅ `losses/__init__.py` - Loss builder
- ✅ `losses/clothes_based_adversarial_loss.py` - Removed distributed gathering

### Models:
- ✅ `models/__init__.py` - Model builder

### Utilities:
- ✅ `tools/utils.py` - General utilities without distributed code

---

## Files to Copy (No Changes Needed)

These files work as-is and just need to be copied:

### Data:
- `data/img_transforms.py`
- `data/spatial_transforms.py`
- `data/temporal_transforms.py`
- `data/datasets/*.py` (all dataset loaders)

### Models:
- `models/classifier.py`
- `models/img_resnet.py`
- `models/vid_resnet.py`
- `models/utils/*.py`

### Losses:
- `losses/arcface_loss.py`
- `losses/circle_loss.py`
- `losses/contrastive_loss.py`
- `losses/cosface_loss.py`
- `losses/cross_entropy_loss_with_label_smooth.py`
- `losses/triplet_loss.py`

### Tools:
- `tools/eval_metrics.py`

### Other:
- `LICENSE`

---

## Troubleshooting

### Missing Files Error
If you get import errors, make sure you've copied all files listed above.

### CUDA Errors
Make sure you have PyTorch with CUDA support installed:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Errors
Reduce batch size in config files if you run out of GPU memory.

---

## Next Steps

Once setup is complete, refer to `README.md` for training and evaluation instructions.
