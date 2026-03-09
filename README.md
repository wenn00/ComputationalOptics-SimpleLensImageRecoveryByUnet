# Residual UNet Image Deblurring

AI-based image deblurring using a Residual UNet architecture. Pretrained on the GOPRO_Large dataset, with support for fine-tuning on custom Raspberry Pi camera data.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Monitoring Training Progress](#monitoring-training-progress)
- [Inference](#inference)
- [Fine-tuning with Custom Data (Future)](#fine-tuning-with-custom-data-future)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project trains a **Residual UNet** neural network to remove motion blur from images. The model learns the mapping from blurry images to sharp images using paired training data from the GOPRO_Large dataset.

**Two-phase strategy:**
1. **Pretrain** on GOPRO_Large (~2100 blur/sharp pairs from high-speed camera footage)
2. **Fine-tune** on custom Raspberry Pi camera data (future, when paired data is collected)

**Supported hardware:**
- NVIDIA GPU with CUDA (recommended: RTX 3080 or above for full training)
- Apple Silicon Mac with MPS (suitable for development and testing)
- CPU (very slow, not recommended for training)

---

## Environment Setup

### Step 1: Set up a Python environment

Use any Python environment manager you prefer (conda, venv, etc.). For reference, here is how to set up with conda:

```bash
conda create -n EECS_195_FinalProject python=3.10
conda activate EECS_195_FinalProject
```

### Step 2: Install PyTorch

PyTorch must be installed separately because the installation command differs by platform.

**NVIDIA GPU (CUDA) — for RTX 3080 or similar:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon Mac (MPS) — for M1/M2/M3 Macs:**

```bash
pip install torch torchvision
```

**CPU only (not recommended for training):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install project dependencies

```bash
pip install -r requirements.txt
```

This installs: `pillow`, `pytorch-msssim`, `tqdm`, `tensorboard`, `numpy`, `pyyaml`, `matplotlib`.

### Step 4: Verify installation

Run a quick check to make sure everything is installed correctly:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

You should see something like:
- On NVIDIA GPU: `PyTorch 2.x.x, CUDA: True, MPS: False`
- On Mac: `PyTorch 2.x.x, CUDA: False, MPS: True`

---

## Dataset Preparation

### Download GOPRO_Large dataset

The GOPRO_Large dataset (~8.9GB) contains paired blurry/sharp images captured from a high-speed camera.

**Download link (Google Drive):**
https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view

### Place the dataset

After downloading, extract the zip file and place the `GOPRO_Large` folder in the project root directory:

```
<project_root>/
├── GOPRO_Large/           ← Place it here
│   ├── train/             # 22 sequences, ~2100 image pairs
│   │   ├── GOPR0372_07_00/
│   │   │   ├── blur/      # Blurry images (PNG, 1280×720)
│   │   │   └── sharp/     # Corresponding sharp images
│   │   ├── GOPR0372_07_01/
│   │   └── ...
│   └── test/              # 11 sequences, ~1111 image pairs (for final evaluation only)
│       ├── GOPR0384_11_00/
│       └── ...
├── src/
├── configs/
└── ...
```

### Verify the dataset

```bash
ls GOPRO_Large/train/ | wc -l    # Should print 22
ls GOPRO_Large/test/ | wc -l     # Should print 11
```

---

## Training

### Quick test (2-3 epochs, verify everything works)

Before running full training, do a quick test to make sure the pipeline works:

1. Open `configs/default.yaml` and temporarily change:
   ```yaml
   epochs: 3
   checkpoint_interval: 1
   ```

2. Run training:
   ```bash
   conda activate EECS_195_FinalProject
   python src/train.py
   ```

3. You should see output like:
   ```
   Using device: mps          (or cuda)
   Train pairs: 1903, Val pairs: 200
   Model parameters: 19,247,427

   Epoch 1/3: 100%|██████████| 237/237 [02:48, loss=0.0207]
   Epoch 1 | Train Loss: 0.0375 | Val PSNR: 24.45 | Val SSIM: 0.8041
     -> New best PSNR: 24.45, saved best.pth
   ```

4. After verifying, change the config back for full training:
   ```yaml
   epochs: 200
   checkpoint_interval: 10
   ```

### Full training

```bash
conda activate EECS_195_FinalProject
python src/train.py
```

To use a custom config file:

```bash
python src/train.py --config configs/default.yaml
```

**What happens during training:**
- The model trains for up to 200 epochs on the GOPRO_Large training set (20 sequences)
- After each epoch, validation metrics (PSNR, SSIM) are computed on 2 held-out sequences
- The best model (highest Val PSNR) is saved to `checkpoints/best.pth`
- A checkpoint is saved every 10 epochs to `checkpoints/epoch_XX.pth`
- Training automatically stops if Val PSNR doesn't improve for 20 consecutive epochs (early stopping)
- Every 10 epochs, training curves and sample comparison images are saved to `results/`

**Estimated training time:**
- RTX 3080: ~1-2 min/epoch → full training ~3-6 hours
- Mac MPS: ~3 min/epoch → full training ~10 hours
- CPU: not recommended (very slow)

**If you get an out-of-memory (OOM) error:**

Open `configs/default.yaml` and reduce `batch_size`, increase `grad_accumulation` to compensate:

```yaml
batch_size: 4              # Reduced from 8
grad_accumulation: 2       # Effective batch = 4 × 2 = 8
```

---

## Monitoring Training Progress

### Training log output

During training, each epoch prints a summary line:

```
Epoch 10 | Train Loss: 0.0285 | Val PSNR: 26.32 | Val SSIM: 0.8451
```

| Metric | What it means | Good direction |
|--------|---------------|----------------|
| **Train Loss** | How wrong the model is on training data | Lower is better ↓ |
| **Val PSNR** | Image reconstruction quality on unseen data (dB) | Higher is better ↑ |
| **Val SSIM** | Structural similarity on unseen data (0-1) | Higher is better ↑ |

**Baseline (no model, blur vs. sharp directly):** PSNR = 22.01 dB, SSIM = 0.6938

**Target after full training:** PSNR ~28-30 dB, SSIM ~0.86-0.92

### Auto-generated plots and images

Every 10 epochs (and at the end of training), the following are saved automatically:

```
results/
├── curves/
│   ├── training_overview.png    # All 3 metrics in one figure
│   ├── loss_curve.png           # Train Loss over epochs
│   ├── psnr_curve.png           # Val PSNR over epochs (with baseline reference line)
│   └── ssim_curve.png           # Val SSIM over epochs (with baseline reference line)
├── samples/
│   ├── epoch_010_sample0.png    # Side-by-side: Blurry | Model Output | Sharp
│   ├── epoch_010_sample1.png
│   ├── epoch_010_sample2.png
│   ├── epoch_020_sample0.png    # Updated every 10 epochs
│   └── ...
```

- **Training curves**: The red dashed line is the baseline (model does nothing). Your curve should be well above it.
- **Sample images**: Three images side by side — Blurry (input) | Model Output | Sharp (ground truth). As training progresses, the middle image should get closer to the right image.

### TensorBoard (optional, real-time monitoring)

Open a separate terminal and run:

```bash
conda activate EECS_195_FinalProject
tensorboard --logdir runs/
```

Then open `http://localhost:6006` in your browser to see live training curves.

---

## Inference

After training, use the best model to deblur new images.

### Deblur a single image

```bash
python src/inference.py --input path/to/blurry.png --output path/to/result.png
```

### Deblur an entire folder

```bash
python src/inference.py --input path/to/blur_folder/ --output path/to/output_folder/
```

### Use a specific checkpoint

By default, it loads `checkpoints/best.pth`. To use a different checkpoint:

```bash
python src/inference.py --input blurry.png --output result.png --checkpoint checkpoints/epoch_100.pth
```

### Large image handling

For images larger than the training patch size (256×256), the inference script automatically:
1. Splits the image into overlapping patches
2. Processes each patch through the model
3. Blends patches together using a Hann window (smooth weighted blending, no seam artifacts)

---

## Fine-tuning with Custom Data (Future)

> **Note:** This step requires paired blur/sharp data from the Raspberry Pi camera, which is not yet available.

### Prepare custom data

Place paired images in the following structure:

```
data/custom/
├── blur/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── sharp/
    ├── 001.png    # Must have matching filenames
    ├── 002.png
    └── ...
```

### Run fine-tuning

```bash
python src/finetune.py --checkpoint checkpoints/best.pth
```

This loads the GOPRO-pretrained model and fine-tunes it on your custom data with a lower learning rate (1e-5) for 50 epochs.

---

## Project Structure

```
├── configs/
│   └── default.yaml         # All hyperparameters and paths
├── src/
│   ├── model.py             # Residual UNet architecture (19.2M parameters)
│   ├── dataset.py           # GOPRO and custom dataset loaders
│   ├── losses.py            # L1 + SSIM loss functions, PSNR/SSIM metrics
│   ├── train.py             # Main training script
│   ├── finetune.py          # Fine-tuning script (for future RPi data)
│   ├── inference.py         # Inference script (single image + batch)
│   └── visualize.py         # Training curve plots and sample image generation
├── GOPRO_Large/             # Dataset (not in git, download separately)
├── checkpoints/             # Saved model weights (not in git)
├── results/
│   ├── curves/              # Training metric plots (auto-generated)
│   └── samples/             # Sample comparison images (auto-generated)
├── runs/                    # TensorBoard logs (not in git)
├── data/custom/             # Custom RPi paired data (future)
├── requirements.txt         # Python dependencies (excluding PyTorch)
├── .gitignore
└── README.md
```

---

## Configuration Reference

All hyperparameters are in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.gopro_root` | `./GOPRO_Large` | Path to GOPRO_Large dataset |
| `data.val_sequences` | `[GOPR0871_11_01, GOPR0881_11_00]` | Fixed validation sequences (2 out of 22) |
| `data.patch_size` | `256` | Random crop size for training |
| `training.batch_size` | `8` | Images per batch (reduce if OOM) |
| `training.grad_accumulation` | `1` | Gradient accumulation steps (increase if reducing batch_size) |
| `training.lr` | `0.0001` | Learning rate (Adam optimizer) |
| `training.epochs` | `200` | Maximum training epochs |
| `training.early_stop_patience` | `20` | Stop if no PSNR improvement for N epochs |
| `training.use_ssim_loss` | `false` | Whether to add SSIM to the loss function |
| `training.ssim_weight` | `0.1` | Weight of SSIM loss when enabled |
| `training.seed` | `42` | Random seed for reproducibility |
| `model.channels` | `[64, 128, 256, 512]` | Channel counts for each encoder/decoder level |
| `model.bottleneck` | `512` | Channel count for the bottleneck layer |
| `save.checkpoint_interval` | `10` | Save checkpoint + plots every N epochs |
| `save.log_dir` | `./runs` | TensorBoard log directory |
| `device` | `auto` | Compute device: `auto` → CUDA → MPS → CPU |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch first (see [Step 2](#step-2-install-pytorch)) |
| `RuntimeError: CUDA out of memory` | Reduce `batch_size` in config, increase `grad_accumulation` |
| `FileNotFoundError: ... GOPRO_Large ...` | Download and place dataset in project root (see [Dataset Preparation](#dataset-preparation)) |
| `pin_memory` warning on Mac | Already handled — `pin_memory` is auto-disabled on MPS |
| Training is very slow on Mac | Normal. Use Mac for testing (2-3 epochs), RTX 3080 for full training |
| Val PSNR stuck and not improving | Training will auto-stop after 20 epochs of no improvement (early stopping) |
| `Loss: nan` or loss explodes | Try reducing `lr` to `0.00005` in config |
