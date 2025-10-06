# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Copy Required Files (2 minutes)

From the `WaitingZone` directory, run:

```bash
python simple-ccReID-refactored/setup_copy_files.py
```

This will copy all necessary files from `Simple-CCReID` to `simple-ccReID-refactored`.

---

### Step 2: Install Dependencies (1 minute)

```bash
cd simple-ccReID-refactored
pip install -r requirements.txt
```

---

### Step 3: Verify Installation (30 seconds)

```bash
python main.py --help
```

You should see the help menu without errors.

---

### Step 4: Prepare Your Dataset (varies)

Download and extract your dataset (e.g., LTCC, PRCC, etc.) to a data directory.

---

### Step 5: Start Training (1 minute to start)

**Example with LTCC dataset:**

```bash
python main.py \
    --dataset ltcc \
    --cfg configs/res50_cels_cal.yaml \
    --root /path/to/your/data \
    --output ./logs \
    --gpu 0
```

**With mixed precision (faster):**

```bash
python main.py \
    --dataset ltcc \
    --cfg configs/res50_cels_cal.yaml \
    --root /path/to/your/data \
    --output ./logs \
    --gpu 0 \
    --amp
```

---

## 📊 Supported Datasets

| Dataset | Command |
|---------|---------|
| **LTCC** | `--dataset ltcc --cfg configs/res50_cels_cal.yaml` |
| **PRCC** | `--dataset prcc --cfg configs/res50_cels_cal.yaml` |
| **VC-Clothes** | `--dataset vcclothes --cfg configs/res50_cels_cal.yaml` |
| **DeepChange** | `--dataset deepchange --cfg configs/res50_cels_cal_16x4.yaml --amp` |
| **LaST** | `--dataset last --cfg configs/res50_cels_cal_tri_16x4.yaml --amp` |
| **CCVID** | `--dataset ccvid --cfg configs/c2dres50_ce_cal.yaml` |

---

## 🎯 Common Use Cases

### Training from Scratch

```bash
python main.py \
    --dataset ltcc \
    --cfg configs/res50_cels_cal.yaml \
    --root /path/to/data \
    --output ./logs \
    --gpu 0
```

### Resume Training

```bash
python main.py \
    --dataset ltcc \
    --cfg configs/res50_cels_cal.yaml \
    --root /path/to/data \
    --output ./logs \
    --gpu 0 \
    --resume ./logs/ltcc/res50-cels-cal/checkpoint_ep50.pth.tar
```

### Evaluation Only

```bash
python main.py \
    --dataset ltcc \
    --cfg configs/res50_cels_cal.yaml \
    --root /path/to/data \
    --output ./logs \
    --gpu 0 \
    --eval \
    --resume ./logs/ltcc/res50-cels-cal/best_model.pth.tar
```

---

## ⚙️ Configuration Tips

### Adjust Batch Size (if out of memory)

Edit the config file or YAML:

```yaml
DATA:
  TRAIN_BATCH: 32  # Reduce from 64 if needed
```

### Change GPU

```bash
python main.py ... --gpu 1  # Use GPU 1 instead of 0
```

### Enable Mixed Precision

```bash
python main.py ... --amp  # Faster training, ~30% speedup
```

---

## 📁 Output Structure

After training, your logs will be organized as:

```
logs/
└── ltcc/                    # Dataset name
    └── res50-cels-cal/      # Experiment tag
        ├── log_train.log    # Training log
        ├── checkpoint_ep5.pth.tar
        ├── checkpoint_ep10.pth.tar
        ├── ...
        └── best_model.pth.tar  # Best performing model
```

---

## 🔍 Monitoring Training

Watch the training log in real-time:

```bash
tail -f logs/ltcc/res50-cels-cal/log_train.log
```

---

## ❓ Troubleshooting

### "CUDA out of memory"
**Solution**: Reduce batch size in config file

### "ModuleNotFoundError"
**Solution**: Run `setup_copy_files.py` to copy missing files

### "No module named 'apex'"
**Solution**: This is expected! The refactored version doesn't use apex

### "FileNotFoundError: [dataset path]"
**Solution**: Check `--root` path points to correct dataset directory

---

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Setup Details**: See `SETUP_INSTRUCTIONS.md`
- **Technical Changes**: See `REFACTORING_SUMMARY.md`
- **Completion Status**: See `COMPLETION_STATUS.md`

---

## 🎉 You're Ready!

That's it! You should now be able to train and evaluate clothes-changing person re-identification models with modern PyTorch on a single GPU.

For more advanced usage and configuration options, check out the full `README.md`.

Happy training! 🚀
