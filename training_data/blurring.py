"""
Image Augmentation Script
Applies Gaussian blur, motion blur, and JPEG compression to images
in Dog, Mobile_phone, and Train folders, saving results to separate
sibling output folders.

folder structure:
  ./Dog/Dog_0001.jpg
  ./Mobile_phone/Mobile_phone_0001.jpg
  ./Train/Train_0001.jpg

output folders:
  ./Dog_Gaussian/          ./Dog_Motion/          ./Dog_Compression/
  ./Mobile_phone_Gaussian/ ./Mobile_phone_Motion/ ./Mobile_phone_Compression/
  ./Train_Gaussian/        ./Train_Motion/        ./Train_Compression/

naming convention:
  Dog_0001_Gaussian.jpg
  Mobile_phone_0001_Motion.jpg
  Train_0001_Compression.jpg
"""

import cv2
import numpy as np
from pathlib import Path

FOLDERS = {
    "Dog":          "Dog",
    "Mobile_phone": "Mobile_phone",
    "Train":        "Train",
}

# gaussian blur
GAUSSIAN_KERNEL = (15, 15)
GAUSSIAN_SIGMA  = 0

# motion blur
MOTION_KERNEL_SIZE = 20

# compression quality
COMPRESSION_QUALITY = 10



def apply_gaussian(image: np.ndarray):
    """Applies Gaussian blur using OpenCV."""
    return cv2.GaussianBlur(image, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)


def apply_motion(image: np.ndarray):
    kernel = np.zeros((MOTION_KERNEL_SIZE, MOTION_KERNEL_SIZE), dtype=np.float32)
    kernel[MOTION_KERNEL_SIZE // 2, :] = 1.0 / MOTION_KERNEL_SIZE
    return cv2.filter2D(image, -1, kernel)


def apply_compression(image: np.ndarray):
    """Encode to JPEG at low quality then decode back to an array."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), COMPRESSION_QUALITY]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# Reuse the same functions for all augmentation types by chaining them together
AUGMENTATIONS = {
    "Gaussian":    apply_gaussian,
    "Motion":      apply_motion,
    "Compression": apply_compression,
    "All":         lambda img: apply_compression(apply_motion(apply_gaussian(img))),
}


def process_folder(folder_path: Path):
    """Process all images in the *folder_path* and save augmented copies to sibling folders."""
    base = folder_path.parent   # parent

    # creates one folder per augmentation type, e.g. "Dog_Gaussian", "Mobile_phone_Motion", etc.
    out_dirs = {}
    for aug_name in AUGMENTATIONS:
        out_dir = base / f"{folder_path.name}_{aug_name}"
        out_dir.mkdir(exist_ok=True)
        out_dirs[aug_name] = out_dir

    image_files = sorted(
        f for f in folder_path.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    )

    if not image_files:
        print(f"no images in {folder_path}") # TEMP DEBUG
        return

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        stem = img_path.stem    # "Dog_0001"
        ext  = img_path.suffix  # ".jpg"

        for aug_name, aug_fn in AUGMENTATIONS.items():
            augmented = aug_fn(image)
            out_name  = f"{stem}_{aug_name}{ext}"
            out_path  = out_dirs[aug_name] / out_name
            cv2.imwrite(str(out_path), augmented)
            print(f"{out_dirs[aug_name].name}/{out_name} fin.") # TEMP DEBUG


def main():
    """Main function to process all folders."""
    base = Path(__file__).parent    # base directory

    for label, folder_name in FOLDERS.items():
        folder_path = base / folder_name
        if not folder_path.is_dir():
            continue

        print(f"Processing [{label}] → {folder_path}") # TEMP DEBUG
        process_folder(folder_path)


if __name__ == "__main__":
    main()