"""Visualization utilities for training progress.

Generates:
1. Training curves (Loss, PSNR, SSIM over epochs)
2. Sample comparison images (blur | model output | sharp ground truth)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, works without display
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(history: dict, save_dir: Path):
    """Plot and save training metric curves.

    Generates three plots:
    - Loss curve (train loss per epoch)
    - PSNR curve (validation PSNR per epoch, with baseline reference)
    - SSIM curve (validation SSIM per epoch, with baseline reference)

    Args:
        history: Dict with keys 'train_loss', 'val_psnr', 'val_ssim',
                 each containing a list of values per epoch.
        save_dir: Directory to save the plot images.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # ---- Loss Curve ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # ---- PSNR Curve ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["val_psnr"], "g-", linewidth=2, label="Val PSNR")
    ax.axhline(y=22.01, color="r", linestyle="--", alpha=0.7, label="Baseline (no model): 22.01 dB")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Validation PSNR", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "psnr_curve.png", dpi=150)
    plt.close(fig)

    # ---- SSIM Curve ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["val_ssim"], "m-", linewidth=2, label="Val SSIM")
    ax.axhline(y=0.6938, color="r", linestyle="--", alpha=0.7, label="Baseline (no model): 0.6938")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("SSIM", fontsize=12)
    ax.set_title("Validation SSIM", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "ssim_curve.png", dpi=150)
    plt.close(fig)

    # ---- Combined Overview ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], "b-", linewidth=2)
    axes[0].set_title("Train Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_psnr"], "g-", linewidth=2)
    axes[1].axhline(y=22.01, color="r", linestyle="--", alpha=0.7, label="Baseline")
    axes[1].set_title("Val PSNR (dB)", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_ssim"], "m-", linewidth=2)
    axes[2].axhline(y=0.6938, color="r", linestyle="--", alpha=0.7, label="Baseline")
    axes[2].set_title("Val SSIM", fontsize=13)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Training Overview", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "training_overview.png", dpi=150)
    plt.close(fig)

    print(f"Training curves saved to {save_dir}/")


def save_sample_images(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    save_dir: Path,
    epoch: int,
    num_samples: int = 3,
):
    """Save side-by-side comparison images: blur | model output | sharp.

    Args:
        model: Trained model (will be set to eval mode).
        val_dataset: Validation dataset.
        device: Compute device.
        save_dir: Directory to save sample images.
        epoch: Current epoch number (for filename).
        num_samples: Number of sample images to save.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Pick evenly spaced samples from val set
    indices = np.linspace(0, len(val_dataset) - 1, num_samples, dtype=int)

    for i, idx in enumerate(indices):
        blur, sharp = val_dataset[idx]
        blur_input = blur.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(blur_input).clamp(0, 1).cpu().squeeze(0)

        # Convert tensors to numpy (C, H, W) -> (H, W, C)
        blur_np = blur.permute(1, 2, 0).numpy()
        output_np = output.permute(1, 2, 0).numpy()
        sharp_np = sharp.permute(1, 2, 0).numpy()

        # Create side-by-side figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(blur_np)
        axes[0].set_title("Blurry (Input)", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(output_np)
        axes[1].set_title(f"Model Output (Epoch {epoch})", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(sharp_np)
        axes[2].set_title("Sharp (Ground Truth)", fontsize=13)
        axes[2].axis("off")

        fig.suptitle(f"Epoch {epoch} — Sample {i+1}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_dir / f"epoch_{epoch:03d}_sample{i}.png", dpi=120)
        plt.close(fig)
