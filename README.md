# Simple-CCReID Refactored

### A Refactored Codebase for Clothes-Changing Person Re-identification

This is a refactored version of [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID), updated to work with:
- **Python 3.12+**
- **PyTorch 2.0+** with native AMP (no apex required)
- **Single GPU training** (removed distributed training dependency)
- Modern dependencies and simplified codebase

#### Original Paper
[Clothes-Changing Person Re-identification with RGB Modality Only (CVPR, 2022)](https://arxiv.org/abs/2204.06890)

---

## Key Changes from Original

### 1. **Removed Dependencies**
- ✅ **Removed apex** - Uses PyTorch native AMP (`torch.amp`) instead
- ✅ **Removed distributed training** - Simplified for single GPU usage
- ✅ **Removed custom DataLoader** - Uses standard PyTorch DataLoader

### 2. **Modernized Code**
- Updated for Python 3.12+ compatibility
- Updated for PyTorch 2.0+ with native mixed precision
- Simplified training loop and data loading
- Cleaner configuration management

### 3. **Single GPU Optimized**
- Batch sizes adjusted for single GPU (doubled from 2-GPU setup)
- Removed all distributed training code
- Simplified samplers (removed DistributedRandomIdentitySampler)
- Direct device placement instead of DDP

---

## Requirements

```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- Python >= 3.12
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- yacs
- numpy
- Pillow

---

## Installation

```bash
# Clone or download this repository
cd simple-ccReID-refactored

# Install dependencies
pip install -r requirements.txt

# Copy remaining files from original Simple-CCReID
# You need to manually copy the following directories from the original repo:
# - data/datasets/*.py (dataset loaders)
# - models/*.py (model architectures)
# - models/utils/*.py (model utilities)
# - losses/*.py (loss functions)
# - tools/eval_metrics.py (evaluation metrics)
```

---

## Dataset Preparation

### CCVID Dataset
- [[BaiduYun]](https://pan.baidu.com/s/1W9yjqxS9qxfPUSu76JpE1g) password: q0q2
- [[GoogleDrive]](https://drive.google.com/file/d/1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W/view?usp=sharing)

### Other Datasets
- **LTCC**: Long-Term Cloth-Changing dataset
- **PRCC**: Person Re-identification with Clothes Change dataset
- **VC-Clothes**: Video-based Clothes-Changing dataset
- **LaST**: Large-Scale Spatio-Temporal dataset
- **DeepChange**: Deep Change dataset

Place your datasets in the `./data/` directory or specify the path using `--root`.

---

## Training

### Basic Training (LTCC Dataset)
```bash
python main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0
```

### Training with Mixed Precision (Faster)
```bash
python main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0 --amp
```

### Training on Different Datasets

**PRCC Dataset:**
```bash
python main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0
```

**VC-Clothes Dataset:**
```bash
python main.py --dataset vcclothes --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0
```

**DeepChange Dataset:**
```bash
python main.py --dataset deepchange --cfg configs/res50_cels_cal_16x4.yaml --root /path/to/data --output ./logs --gpu 0 --amp
```

**LaST Dataset:**
```bash
python main.py --dataset last --cfg configs/res50_cels_cal_tri_16x4.yaml --root /path/to/data --output ./logs --gpu 0 --amp
```

**CCVID Dataset (Video):**
```bash
python main.py --dataset ccvid --cfg configs/c2dres50_ce_cal.yaml --root /path/to/data --output ./logs --gpu 0
```

---

## Evaluation

```bash
python main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --root /path/to/data --output ./logs --gpu 0 --eval --resume /path/to/checkpoint.pth.tar
```

---

## Configuration

Configuration files are in `configs/`:
- `res50_cels_cal.yaml` - ResNet50 with Cross-Entropy Label Smoothing + CAL
- `c2dres50_ce_cal.yaml` - C2D ResNet50 for video (CCVID)
- `res50_cels_cal_16x4.yaml` - ResNet50 with 16x4 batch configuration
- `res50_cels_cal_tri_16x4.yaml` - ResNet50 with triplet loss

You can modify `configs/default_img.py` or `configs/default_vid.py` for default settings.

---

## Key Differences in Usage

### Original (2 GPUs with DDP):
```bash
python -m torch.distributed.launch --nproc_per_node=2 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1
```

### Refactored (Single GPU):
```bash
python main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0
```

**Note:** Batch sizes have been doubled in the refactored version to compensate for single GPU training.

---

## Project Structure

```
simple-ccReID-refactored/
├── main.py                 # Main training script (refactored)
├── train.py                # Training functions (refactored)
├── test.py                 # Testing functions (refactored)
├── requirements.txt        # Modern dependencies
├── README.md              # This file
├── configs/               # Configuration files
│   ├── default_img.py     # Default config for image datasets
│   ├── default_vid.py     # Default config for video datasets
│   └── *.yaml            # Experiment-specific configs
├── data/                  # Data loading modules
│   ├── __init__.py        # Data loader builder (refactored)
│   ├── samplers.py        # Simplified samplers (refactored)
│   ├── dataset_loader.py  # Dataset classes
│   ├── img_transforms.py  # Image transformations
│   ├── spatial_transforms.py  # Spatial transformations
│   ├── temporal_transforms.py # Temporal transformations
│   └── datasets/          # Dataset-specific loaders
├── models/                # Model architectures
│   ├── __init__.py        # Model builder
│   ├── classifier.py      # Classifier heads
│   ├── img_resnet.py      # ResNet for images
│   ├── vid_resnet.py      # ResNet for videos
│   └── utils/             # Model utilities
├── losses/                # Loss functions
│   ├── __init__.py        # Loss builder
│   ├── clothes_based_adversarial_loss.py  # CAL loss (refactored)
│   └── *.py              # Other loss functions
└── tools/                 # Utility functions
    ├── utils.py           # General utilities (refactored)
    └── eval_metrics.py    # Evaluation metrics
```

---

## Performance Notes

- **Batch Size:** Doubled compared to original 2-GPU setup to maintain effective batch size
- **Mixed Precision:** Use `--amp` flag for faster training with minimal accuracy loss
- **Memory:** Requires ~11GB GPU memory for default settings (adjust batch size if needed)

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config files:
```yaml
DATA:
  TRAIN_BATCH: 32  # Reduce this value
```

### Import Errors
Make sure all files from original Simple-CCReID are copied:
```bash
# Copy missing files manually from Simple-CCReID/
cp -r ../Simple-CCReID/data/datasets/*.py ./data/datasets/
cp -r ../Simple-CCReID/models/*.py ./models/
cp -r ../Simple-CCReID/losses/*.py ./losses/
cp -r ../Simple-CCReID/tools/eval_metrics.py ./tools/
```

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{gu2022CAL,
    title={Clothes-Changing Person Re-identification with RGB Modality Only},
    author={Gu, Xinqian and Chang, Hong and Ma, Bingpeng and Bai, Shutao and Shan, Shiguang and Chen, Xilin},
    booktitle={CVPR},
    year={2022},
}
```

## License

This project follows the same license as the original Simple-CCReID repository.

---

## Acknowledgments

This refactored version is based on the original [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID) by Xinqian Gu et al. All credit for the methodology and original implementation goes to the original authors.
