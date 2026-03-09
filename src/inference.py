"""Inference script for Residual UNet image deblurring.

Supports single image and batch folder processing.
For large images, uses sliding window with overlap and weighted blending.

Usage:
    python src/inference.py --input path/to/blurry.png --output path/to/output.png
    python src/inference.py --input path/to/blur_folder/ --output path/to/output_folder/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import ResidualUNet
from train import get_device, load_config


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> ResidualUNet:
    """Load trained model from checkpoint."""
    model = ResidualUNet(
        channels=config["model"]["channels"],
        bottleneck=config["model"]["bottleneck"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    psnr = checkpoint.get("best_psnr", "?")
    print(f"Loaded model from epoch {epoch} (PSNR: {psnr})")
    return model


def create_hann_window(size: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Hann window for weighted blending.

    The window has high weight at the center and tapers to near-zero at edges,
    ensuring smooth transitions when overlapping patches are blended.

    Args:
        size: Window size (square).
        device: Compute device.

    Returns:
        (1, 1, size, size) tensor with Hann window weights.
    """
    hann_1d = torch.hann_window(size, periodic=False, device=device)
    hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
    return hann_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def inference_single(
    model: ResidualUNet,
    image: torch.Tensor,
    device: torch.device,
    patch_size: int = 256,
    overlap: int = 64,
) -> torch.Tensor:
    """Run inference on a single image, using sliding window for large images.

    Args:
        model: Trained model in eval mode.
        image: Input tensor (1, 3, H, W) in [0, 1].
        device: Compute device.
        patch_size: Size of each processing patch.
        overlap: Overlap between adjacent patches.

    Returns:
        Deblurred image tensor (1, 3, H, W) clamped to [0, 1].
    """
    _, _, h, w = image.shape

    # If image fits in one patch, process directly
    if h <= patch_size and w <= patch_size:
        with torch.no_grad():
            output = model(image.to(device))
        return output.clamp(0, 1).cpu()

    # Sliding window with weighted blending
    stride = patch_size - overlap
    hann = create_hann_window(patch_size, device)  # (1, 1, ps, ps)

    # Pad image so patches tile evenly
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    padded = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, ph, pw = padded.shape

    output = torch.zeros_like(padded)
    weight = torch.zeros(1, 1, ph, pw)

    for top in range(0, ph - patch_size + 1, stride):
        for left in range(0, pw - patch_size + 1, stride):
            patch = padded[:, :, top : top + patch_size, left : left + patch_size]
            with torch.no_grad():
                pred = model(patch.to(device)).cpu()

            output[:, :, top : top + patch_size, left : left + patch_size] += pred * hann.cpu()
            weight[:, :, top : top + patch_size, left : left + patch_size] += hann.cpu()

    # Normalize by weights and remove padding
    output = output / weight.clamp(min=1e-8)
    output = output[:, :, :h, :w]
    return output.clamp(0, 1)


def process_image(
    model: ResidualUNet,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    patch_size: int = 256,
):
    """Load, deblur, and save a single image."""
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)

    output = inference_single(model, img_tensor, device, patch_size)

    # Convert back to uint8 PIL image
    output_np = (output.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    output_img = Image.fromarray(output_np)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Deblur images with trained model")
    parser.add_argument("--input", type=str, required=True, help="Input image or folder")
    parser.add_argument("--output", type=str, required=True, help="Output image or folder")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "best.pth"),
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size for sliding window")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["device"])
    model = load_model(args.checkpoint, config, device)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single image
        print(f"Processing: {input_path}")
        process_image(model, input_path, output_path, device, args.patch_size)
        print(f"Saved to: {output_path}")
    elif input_path.is_dir():
        # Batch folder
        image_files = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
        if not image_files:
            print(f"No images found in {input_path}")
            return
        output_path.mkdir(parents=True, exist_ok=True)
        for img_file in tqdm(image_files, desc="Deblurring"):
            out_file = output_path / img_file.name
            process_image(model, img_file, out_file, device, args.patch_size)
        print(f"Processed {len(image_files)} images -> {output_path}")
    else:
        print(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
