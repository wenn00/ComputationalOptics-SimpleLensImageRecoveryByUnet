"""Dataset classes for GOPRO_Large and custom paired data.

All images are loaded with PIL (consistent RGB), converted to float32 in [0, 1].
Augmentation: synchronized random crop + horizontal flip for blur/sharp pairs.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_image(path: Path) -> np.ndarray:
    """Load image as float32 numpy array in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def synchronized_crop(blur: np.ndarray, sharp: np.ndarray, patch_size: int):
    """Apply the same random crop to both blur and sharp images.

    Args:
        blur: (H, W, 3) float32 array.
        sharp: (H, W, 3) float32 array.
        patch_size: Size of the square crop.

    Returns:
        Cropped (blur, sharp) tuple.
    """
    h, w = blur.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Image size ({h}x{w}) is smaller than patch_size ({patch_size})"
        )
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    blur = blur[top : top + patch_size, left : left + patch_size]
    sharp = sharp[top : top + patch_size, left : left + patch_size]
    return blur, sharp


def synchronized_hflip(blur: np.ndarray, sharp: np.ndarray, p: float = 0.5):
    """Apply the same random horizontal flip to both images."""
    if random.random() < p:
        blur = np.flip(blur, axis=1).copy()
        sharp = np.flip(sharp, axis=1).copy()
    return blur, sharp


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert (H, W, C) numpy array to (C, H, W) torch tensor."""
    return torch.from_numpy(img.transpose(2, 0, 1))


class GoproDataset(Dataset):
    """GOPRO_Large dataset for image deblurring.

    Scans sequences under train/*/blur/ and train/*/sharp/.
    Uses sorted filenames for correct blur-sharp pairing.

    Args:
        root: Path to GOPRO_Large directory.
        sequences: List of sequence names to include.
        patch_size: Random crop size (None for full images).
        augment: Whether to apply augmentation (crop + flip).
    """

    def __init__(
        self,
        root: str | Path,
        sequences: list[str],
        patch_size: int | None = 256,
        augment: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.patch_size = patch_size
        self.augment = augment
        self.pairs: list[tuple[Path, Path]] = []

        for seq in sequences:
            blur_dir = self.root / "train" / seq / "blur"
            sharp_dir = self.root / "train" / seq / "sharp"

            if not blur_dir.exists():
                raise FileNotFoundError(f"Blur directory not found: {blur_dir}")
            if not sharp_dir.exists():
                raise FileNotFoundError(f"Sharp directory not found: {sharp_dir}")

            blur_files = sorted(blur_dir.glob("*.png"))
            sharp_files = sorted(sharp_dir.glob("*.png"))

            if len(blur_files) != len(sharp_files):
                raise ValueError(
                    f"Sequence {seq}: blur ({len(blur_files)}) and sharp "
                    f"({len(sharp_files)}) have different number of images"
                )

            for b, s in zip(blur_files, sharp_files):
                if b.stem != s.stem:
                    raise ValueError(
                        f"Filename mismatch in {seq}: {b.name} vs {s.name}"
                    )
                self.pairs.append((b, s))

        if len(self.pairs) == 0:
            raise ValueError("No image pairs found. Check dataset path and sequences.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        blur_path, sharp_path = self.pairs[idx]
        blur = load_image(blur_path)
        sharp = load_image(sharp_path)

        if self.augment and self.patch_size is not None:
            blur, sharp = synchronized_crop(blur, sharp, self.patch_size)
            blur, sharp = synchronized_hflip(blur, sharp)

        return to_tensor(blur), to_tensor(sharp)


class CustomDataset(Dataset):
    """Custom paired dataset for fine-tuning (e.g., RPi camera data).

    Expects directory structure:
        root/blur/*.png
        root/sharp/*.png

    Args:
        root: Path to custom data directory.
        patch_size: Random crop size (None for full images).
        augment: Whether to apply augmentation.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int | None = 256,
        augment: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.patch_size = patch_size
        self.augment = augment
        self.pairs: list[tuple[Path, Path]] = []

        blur_dir = self.root / "blur"
        sharp_dir = self.root / "sharp"

        if not blur_dir.exists() or not sharp_dir.exists():
            raise FileNotFoundError(
                f"Expected blur/ and sharp/ subdirectories in {self.root}"
            )

        blur_files = sorted(blur_dir.glob("*.png"))
        sharp_files = sorted(sharp_dir.glob("*.png"))

        if len(blur_files) != len(sharp_files):
            raise ValueError(
                f"blur ({len(blur_files)}) and sharp ({len(sharp_files)}) "
                f"have different number of images"
            )

        for b, s in zip(blur_files, sharp_files):
            if b.stem != s.stem:
                raise ValueError(f"Filename mismatch: {b.name} vs {s.name}")
            self.pairs.append((b, s))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        blur_path, sharp_path = self.pairs[idx]
        blur = load_image(blur_path)
        sharp = load_image(sharp_path)

        if self.augment and self.patch_size is not None:
            blur, sharp = synchronized_crop(blur, sharp, self.patch_size)
            blur, sharp = synchronized_hflip(blur, sharp)

        return to_tensor(blur), to_tensor(sharp)


def get_gopro_splits(
    gopro_root: str | Path, val_sequences: list[str], patch_size: int = 256
) -> tuple[GoproDataset, GoproDataset]:
    """Create train and val datasets from GOPRO_Large with sequence-level split.

    Args:
        gopro_root: Path to GOPRO_Large directory.
        val_sequences: List of sequence names for validation.
        patch_size: Random crop size for training.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    gopro_root = Path(gopro_root)
    all_sequences = sorted(
        [d.name for d in (gopro_root / "train").iterdir() if d.is_dir()]
    )

    train_sequences = [s for s in all_sequences if s not in val_sequences]
    for vs in val_sequences:
        if vs not in all_sequences:
            raise ValueError(f"Val sequence '{vs}' not found in {gopro_root}/train/")

    print(f"Train sequences ({len(train_sequences)}): {train_sequences}")
    print(f"Val sequences ({len(val_sequences)}): {val_sequences}")

    train_ds = GoproDataset(
        gopro_root, train_sequences, patch_size=patch_size, augment=True
    )
    val_ds = GoproDataset(
        gopro_root, val_sequences, patch_size=None, augment=False
    )

    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")
    return train_ds, val_ds
