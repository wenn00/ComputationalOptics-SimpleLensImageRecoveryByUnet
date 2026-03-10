"""Fine-tuning script for Residual UNet on custom RPi paired data.

Loads a pretrained checkpoint and fine-tunes on custom paired data with
90/10 train/val split. Tracks validation PSNR/SSIM and saves best model.

Usage:
    python src/finetune.py --checkpoint checkpoints/best.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import CustomDataset
from inference import inference_single
from losses import CombinedLoss, compute_psnr, compute_ssim
from model import ResidualUNet
from train import get_device, load_config, set_seed


def save_comparison_images(
    model: ResidualUNet,
    val_dataset: CustomDataset,
    val_indices: list[int],
    device: torch.device,
    output_dir: Path,
    num_samples: int = 5,
):
    """Generate side-by-side comparison images (Blurry | Deblurred | Sharp GT).

    Picks evenly-spaced samples from the validation set, runs inference with
    the best model, and saves labeled comparison images.

    Args:
        model: Trained model (already loaded with best weights).
        val_dataset: Dataset with patch_size=None, augment=False.
        val_indices: Indices of validation samples.
        device: Compute device.
        output_dir: Directory to save comparison PNGs.
        num_samples: Number of comparison images to generate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Pick evenly-spaced samples
    n = len(val_indices)
    num_samples = min(num_samples, n)
    step = max(1, n // num_samples)
    selected = [val_indices[i * step] for i in range(num_samples)]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()

    for i, idx in enumerate(selected):
        blur_tensor, sharp_tensor = val_dataset[idx]  # (3, H, W)

        # Run inference (sliding window for large images)
        blur_input = blur_tensor.unsqueeze(0)  # (1, 3, H, W)
        with torch.no_grad():
            output = inference_single(model, blur_input, device, patch_size=256)
        output = output.squeeze(0)  # (3, H, W)

        # Convert tensors to PIL images
        to_pil = lambda t: Image.fromarray(
            (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        blur_img = to_pil(blur_tensor)
        output_img = to_pil(output)
        sharp_img = to_pil(sharp_tensor)

        # Create labeled side-by-side comparison
        w, h = blur_img.size
        label_h = 30
        comparison = Image.new("RGB", (w * 3, h + label_h), (255, 255, 255))
        comparison.paste(blur_img, (0, label_h))
        comparison.paste(output_img, (w, label_h))
        comparison.paste(sharp_img, (w * 2, label_h))

        draw = ImageDraw.Draw(comparison)
        draw.text((w // 2 - 30, 5), "Blurry", fill=(255, 0, 0), font=font)
        draw.text((w + w // 2 - 40, 5), "Deblurred", fill=(0, 128, 0), font=font)
        draw.text((w * 2 + w // 2 - 50, 5), "Sharp (GT)", fill=(0, 0, 255), font=font)

        comparison.save(output_dir / f"comparison_{i + 1}.png")

    print(f"Saved {num_samples} comparison images to {output_dir}/")



def finetune(config: dict, checkpoint_path: str):
    """Fine-tune a pretrained model on custom data."""
    seed = config["training"]["seed"]
    set_seed(seed)
    device = get_device(config["device"])
    print(f"Using device: {device}")

    # ---- Load pretrained model ----
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ResidualUNet(
        channels=config["model"]["channels"],
        bottleneck=config["model"]["bottleneck"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(PSNR: {checkpoint['best_psnr']:.2f})")

    # ---- Custom dataset with train/val split ----
    custom_root = PROJECT_ROOT / config["data"]["custom_root"]
    patch_size = config["data"]["patch_size"]

    # Build index list, then split into train/val indices
    # We create two datasets: one with augmentation (train), one without (val)
    train_dataset = CustomDataset(custom_root, patch_size=patch_size, augment=True)
    val_dataset = CustomDataset(custom_root, patch_size=None, augment=False)

    n_total = len(train_dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val

    # Deterministic split using seeded generator
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,  # Full-resolution val images
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory,
    )
    print(f"Custom dataset: {n_total} pairs (train: {n_train}, val: {n_val})")

    # ---- Fine-tune setup ----
    finetune_lr = 1e-5
    finetune_epochs = 50
    criterion = CombinedLoss(use_ssim=True, ssim_weight=0.1)
    optimizer = Adam(model.parameters(), lr=finetune_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    best_psnr = 0.0

    # ---- Training loop ----
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Finetune {epoch}/{finetune_epochs}")
        for blur, sharp in pbar:
            blur = blur.to(device)
            sharp = sharp.to(device)

            output = model(blur)
            loss = criterion(output, sharp)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_loss_sum = 0.0

        with torch.no_grad():
            for blur, sharp in val_loader:
                blur = blur.to(device)
                sharp = sharp.to(device)

                output = model(blur).clamp(0, 1)
                val_loss_sum += criterion(output, sharp).item()
                val_psnr_sum += compute_psnr(output, sharp)
                val_ssim_sum += compute_ssim(output, sharp)

        n_val_batches = len(val_loader)
        val_psnr = val_psnr_sum / n_val_batches
        val_ssim = val_ssim_sum / n_val_batches

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")

        # ---- Save best model ----
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_psnr": best_psnr,
                    "config": config,
                },
                checkpoint_dir / "finetune_best.pth",
            )
            print(f"  -> New best PSNR: {best_psnr:.2f}, saved finetune_best.pth")

        # ---- Periodic checkpoint ----
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"finetune_epoch_{epoch}.pth",
            )

    # Save final
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        checkpoint_dir / "finetune_final.pth",
    )
    print(f"Fine-tuning complete. Best val PSNR: {best_psnr:.2f}")

    # ---- Generate comparison images with best model ----
    best_ckpt = torch.load(
        checkpoint_dir / "finetune_best.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("Loaded best model for generating comparison images...")

    results_dir = PROJECT_ROOT / "results" / "finetune_samples"
    save_comparison_images(model, val_dataset, val_indices, device, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune on custom RPi data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Pretrained checkpoint path")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    finetune(config, args.checkpoint)
