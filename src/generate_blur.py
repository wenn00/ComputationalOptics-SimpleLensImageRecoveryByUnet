"""Generate synthetic Gaussian blur images for fine-tuning.

Reads sharp images from data/custom/sharp/, applies random Gaussian blur
with optional noise, and saves to data/custom/blur/ with matching filenames.

Usage:
    python src/generate_blur.py --sigma_min 1.0 --sigma_max 5.0 --seed 42
    python src/generate_blur.py --no-noise  # disable additive noise
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def apply_blur(image: Image.Image, sigma: float, add_noise: bool,
               noise_sigma: float, rng: np.random.Generator) -> Image.Image:
    """Apply Gaussian blur (and optional noise) to a PIL image.

    This is the core degradation function. It determines how synthetic
    blurry images are generated from sharp originals.

    TODO: Implement this function (~5-10 lines). Consider:
    - Compute kernel_size from sigma: 2 * ceil(3 * sigma) + 1
    - Apply Gaussian blur using PIL's ImageFilter.GaussianBlur(radius)
      Note: PIL's radius parameter = kernel_size // 2
    - Optionally add Gaussian noise (convert to numpy, add noise, convert back)
    - Clamp pixel values to [0, 255] after adding noise

    Args:
        image: Input PIL Image (RGB, uint8).
        sigma: Gaussian blur standard deviation.
        add_noise: Whether to add Gaussian noise after blurring.
        noise_sigma: Std of additive Gaussian noise (in [0, 1] scale).
        rng: NumPy random generator for reproducible noise.

    Returns:
        Degraded PIL Image (RGB, uint8).
    """
    # Step 1: Compute kernel radius from sigma, then apply Gaussian blur
    radius = math.ceil(3 * sigma)  # kernel_size = 2*radius+1 (always odd)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # Step 2: Optionally add light Gaussian noise for realism
    if add_noise:
        arr = np.array(blurred, dtype=np.float32) / 255.0
        noise = rng.normal(0, noise_sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        blurred = Image.fromarray((arr * 255).astype(np.uint8))

    return blurred


def generate_blur_dataset(
    sharp_dir: Path,
    blur_dir: Path,
    sigma_min: float,
    sigma_max: float,
    add_noise: bool,
    noise_sigma: float,
    seed: int,
):
    """Generate blurred versions of all sharp images.

    Args:
        sharp_dir: Directory containing sharp PNG images.
        blur_dir: Output directory for blurred images.
        sigma_min: Minimum blur sigma.
        sigma_max: Maximum blur sigma.
        add_noise: Whether to add Gaussian noise.
        noise_sigma: Noise standard deviation (in [0, 1] scale).
        seed: Random seed for reproducibility.
    """
    blur_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    image_paths = sorted(sharp_dir.glob("*.png"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No PNG images found in {sharp_dir}")

    print(f"Found {len(image_paths)} sharp images in {sharp_dir}")
    print(f"Blur sigma range: [{sigma_min}, {sigma_max}]")
    print(f"Noise: {'on (sigma={:.4f})'.format(noise_sigma) if add_noise else 'off'}")
    print(f"Output directory: {blur_dir}")

    for img_path in tqdm(image_paths, desc="Generating blur"):
        image = Image.open(img_path).convert("RGB")
        sigma = rng.uniform(sigma_min, sigma_max)
        blurred = apply_blur(image, sigma, add_noise, noise_sigma, rng)
        blurred.save(blur_dir / img_path.name)

    print(f"Done. Generated {len(image_paths)} blurred images.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Gaussian blur images"
    )
    parser.add_argument(
        "--sharp_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "custom" / "sharp"),
        help="Directory containing sharp images",
    )
    parser.add_argument(
        "--blur_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "custom" / "blur"),
        help="Output directory for blurred images",
    )
    parser.add_argument("--sigma_min", type=float, default=1.0,
                        help="Minimum Gaussian blur sigma (default: 1.0)")
    parser.add_argument("--sigma_max", type=float, default=5.0,
                        help="Maximum Gaussian blur sigma (default: 5.0)")
    parser.add_argument("--noise", action=argparse.BooleanOptionalAction,
                        default=True, help="Add Gaussian noise (default: on)")
    parser.add_argument("--noise_sigma", type=float, default=2.0 / 255.0,
                        help="Noise std in [0,1] scale (default: 2/255)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_blur_dataset(
        sharp_dir=Path(args.sharp_dir),
        blur_dir=Path(args.blur_dir),
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        add_noise=args.noise,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
