"""Training script for Residual UNet on GOPRO_Large.

Usage:
    python src/train.py
    python src/train.py --config configs/default.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to path for imports
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import get_gopro_splits
from losses import CombinedLoss, compute_psnr, compute_ssim
from model import ResidualUNet
from visualize import plot_training_curves, save_sample_images


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_config: str) -> torch.device:
    """Auto-detect best available device: CUDA -> MPS -> CPU."""
    if device_config != "auto":
        return torch.device(device_config)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate(
    model: ResidualUNet,
    val_loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run validation and compute metrics.

    TODO: Implement the validation evaluation logic.

    This function should:
    1. Set model to eval mode
    2. Iterate over val_loader with torch.no_grad()
    3. For each batch:
       - Move blur/sharp to device
       - Forward pass through model
       - Clamp output to [0, 1] (important for correct PSNR/SSIM)
       - Accumulate loss, PSNR, and SSIM
    4. Return averaged (val_loss, val_psnr, val_ssim)

    Args:
        model: The trained model.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (avg_loss, avg_psnr, avg_ssim).
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for blur, sharp in val_loader:
            blur = blur.to(device)
            sharp = sharp.to(device)

            output = model(blur)
            output = output.clamp(0, 1)  # Clamp before metrics

            total_loss += criterion(output, sharp).item()
            total_psnr += compute_psnr(output, sharp)
            total_ssim += compute_ssim(output, sharp)

    n = len(val_loader)
    return total_loss / n, total_psnr / n, total_ssim / n


def train(config: dict):
    """Main training loop."""
    # ---- Setup ----
    set_seed(config["training"]["seed"])
    device = get_device(config["device"])
    print(f"Using device: {device}")

    # ---- Data ----
    gopro_root = PROJECT_ROOT / config["data"]["gopro_root"]
    train_ds, val_ds = get_gopro_splits(
        gopro_root,
        config["data"]["val_sequences"],
        config["data"]["patch_size"],
    )

    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Full-resolution val images
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    # ---- Model ----
    model = ResidualUNet(
        channels=config["model"]["channels"],
        bottleneck=config["model"]["bottleneck"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # ---- Loss, Optimizer, Scheduler ----
    criterion = CombinedLoss(
        use_ssim=config["training"]["use_ssim_loss"],
        ssim_weight=config["training"]["ssim_weight"],
    )
    optimizer = Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # ---- Logging ----
    log_dir = PROJECT_ROOT / config["save"]["log_dir"]
    writer = SummaryWriter(log_dir)
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ---- Training Loop ----
    best_psnr = 0.0
    patience_counter = 0
    grad_accum = config["training"]["grad_accumulation"]
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stop_patience"]
    ckpt_interval = config["save"]["checkpoint_interval"]
    results_dir = PROJECT_ROOT / "results"

    # Track metrics for plotting
    history = {"train_loss": [], "val_psnr": [], "val_ssim": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        step = 0
        for step, (blur, sharp) in enumerate(pbar):
            blur = blur.to(device)
            sharp = sharp.to(device)

            output = model(blur)
            loss = criterion(output, sharp) / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        # Handle remaining gradients
        if (step + 1) % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)

        # ---- Logging ----
        history["train_loss"].append(avg_train_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_psnr", val_psnr, epoch)
        writer.add_scalar("Metrics/val_ssim", val_ssim, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}"
        )

        # ---- Checkpointing ----
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "config": config,
                },
                checkpoint_dir / "best.pth",
            )
            print(f"  -> New best PSNR: {best_psnr:.2f}, saved best.pth")
        else:
            patience_counter += 1

        if epoch % ckpt_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "config": config,
                },
                checkpoint_dir / f"epoch_{epoch}.pth",
            )
            # Save sample comparison images
            save_sample_images(
                model, val_ds, device, results_dir / "samples", epoch
            )
            # Update training curves
            plot_training_curves(history, results_dir / "curves")

        # ---- Early Stopping ----
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    writer.close()

    # ---- Generate final plots ----
    plot_training_curves(history, results_dir / "curves")
    save_sample_images(model, val_ds, device, results_dir / "samples", epoch)

    print(f"Training complete. Best val PSNR: {best_psnr:.2f}")
    print(f"Curves saved to: {results_dir / 'curves'}/")
    print(f"Sample images saved to: {results_dir / 'samples'}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Residual UNet on GOPRO_Large")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
