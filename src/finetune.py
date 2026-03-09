"""Fine-tuning script for Residual UNet on custom RPi paired data.

This script will be fully implemented when RPi camera data is available.
It loads a pretrained checkpoint and fine-tunes on custom paired data.

Usage:
    python src/finetune.py --checkpoint checkpoints/best.pth
"""

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import CustomDataset
from losses import CombinedLoss
from model import ResidualUNet
from train import get_device, load_config, set_seed


def finetune(config: dict, checkpoint_path: str):
    """Fine-tune a pretrained model on custom data."""
    set_seed(config["training"]["seed"])
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

    # ---- Custom dataset ----
    custom_root = PROJECT_ROOT / config["data"]["custom_root"]
    dataset = CustomDataset(custom_root, patch_size=config["data"]["patch_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Custom dataset: {len(dataset)} pairs")

    # ---- Fine-tune setup ----
    finetune_lr = 1e-5  # Lower lr for fine-tuning
    finetune_epochs = 50
    criterion = CombinedLoss(use_ssim=True, ssim_weight=0.1)
    optimizer = Adam(model.parameters(), lr=finetune_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ---- Training loop ----
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Finetune {epoch}/{finetune_epochs}")
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
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

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
    print("Fine-tuning complete.")


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
