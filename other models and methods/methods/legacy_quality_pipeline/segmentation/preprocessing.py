"""
Image preprocessing and validation module.
Handles EXIF rotation, format conversion, decompression bomb protection,
and conversion to the required tensor formats for ML models.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags, ImageFile

logger = logging.getLogger(__name__)

# Allow loading truncated images but with warning
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Decompression bomb limit (100 MP)
MAX_IMAGE_PIXELS = 100_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}


class ImageValidationError(Exception):
    """Raised when an image fails validation."""
    pass


class ImageInfo:
    """Metadata about a loaded image."""
    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
        megapixels: float,
        format_: str,
        mode: str,
        has_exif: bool,
        exif_rotation: Optional[int],
        original_path: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.channels = channels
        self.megapixels = megapixels
        self.format_ = format_
        self.mode = mode
        self.has_exif = has_exif
        self.exif_rotation = exif_rotation
        self.original_path = original_path

    def to_dict(self) -> Dict:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "megapixels": round(self.megapixels, 2),
            "format": self.format_,
            "mode": self.mode,
            "has_exif": self.has_exif,
            "exif_rotation": self.exif_rotation,
            "original_path": self.original_path,
        }


def validate_image_bytes(data: bytes, filename: str) -> None:
    """
    Validate raw image bytes before loading.
    
    Raises ImageValidationError if invalid.
    """
    if not data or len(data) < 8:
        raise ImageValidationError("File is empty or too small to be a valid image.")
    
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def load_image(
    source: str | bytes,
    filename: str = "image.jpg",
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> Tuple[np.ndarray, ImageInfo]:
    """
    Load an image from file path or bytes, with full preprocessing.
    
    - Handles EXIF rotation
    - Converts to RGB (preserving colour info)
    - Handles RGBA (transparent PNGs)
    - Handles grayscale
    - Protects against decompression bombs
    
    Returns:
        (image_rgb, image_info) — image is always RGB uint8 numpy array
    """
    if isinstance(source, (str, Path)):
        source = str(source)
        try:
            pil_img = Image.open(source)
        except Exception as e:
            raise ImageValidationError(f"Cannot decode image: {e}")
        filename = source
    elif isinstance(source, bytes):
        validate_image_bytes(source, filename)
        try:
            pil_img = Image.open(io.BytesIO(source))
        except Exception as e:
            raise ImageValidationError(f"Cannot decode image: {e}")
    else:
        raise ImageValidationError("Source must be a file path or bytes.")
    
    # Check for decompression bomb
    width, height = pil_img.size
    total_pixels = width * height
    if total_pixels > max_pixels:
        raise ImageValidationError(
            f"Image is too large ({width}x{height} = {total_pixels / 1e6:.1f} MP). "
            f"Maximum allowed: {max_pixels / 1e6:.0f} MP. "
            "This is a memory safety limit, not a quality rejection."
        )
    
    # Get EXIF info
    has_exif = False
    exif_rotation = None
    try:
        exif_data = pil_img._getexif()
        if exif_data:
            has_exif = True
            orientation_key = None
            for tag, name in ExifTags.TAGS.items():
                if name == "Orientation":
                    orientation_key = tag
                    break
            if orientation_key and orientation_key in exif_data:
                exif_rotation = exif_data[orientation_key]
    except (AttributeError, Exception):
        pass
    
    # Apply EXIF orientation
    try:
        from PIL import ImageOps
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    
    format_ = pil_img.format or Path(filename).suffix.lstrip(".").upper()
    original_mode = pil_img.mode
    
    # Convert to RGB, preserving colour information
    if pil_img.mode == "RGBA":
        # Composite onto white background for transparent PNGs
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode == "L":
        # Grayscale -> RGB (3-channel for consistent processing)
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode == "P":
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode == "I;16":
        # 16-bit TIFF — convert carefully
        arr = np.array(pil_img, dtype=np.uint16)
        arr = (arr / 256).astype(np.uint8)
        pil_img = Image.fromarray(arr).convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    
    # To numpy RGB
    img_rgb = np.array(pil_img, dtype=np.uint8)
    
    width, height = pil_img.size
    channels = img_rgb.shape[2] if img_rgb.ndim == 3 else 1
    megapixels = (width * height) / 1_000_000
    
    info = ImageInfo(
        width=width,
        height=height,
        channels=channels,
        megapixels=megapixels,
        format_=format_,
        mode=original_mode,
        has_exif=has_exif,
        exif_rotation=exif_rotation,
        original_path=filename if isinstance(filename, str) else None,
    )
    
    logger.info(
        f"Loaded image: {width}x{height} ({megapixels:.2f} MP), "
        f"mode={original_mode}, format={format_}"
    )
    
    return img_rgb, info


def resize_for_inference(
    image: np.ndarray,
    max_size: int = 1024,
) -> Tuple[np.ndarray, float]:
    """
    Resize image for inference while preserving aspect ratio.
    Returns (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim <= max_size:
        return image.copy(), 1.0
    
    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info(f"Resized for inference: {w}x{h} -> {new_w}x{new_h} (scale={scale:.4f})")
    return resized, scale


def pad_to_square(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Pad image to square preserving aspect ratio, then resize.
    Used for CNN inputs (VGG-19, ResNet-18) to avoid stretching rice grain crops.
    
    Args:
        image: Input image (H, W, C)
        target_size: Output square dimension
        
    Returns:
        Square padded image of shape (target_size, target_size, C)
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    # Create square canvas (black padding)
    if image.ndim == 3:
        padded = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((max_dim, max_dim), dtype=image.dtype)
    
    # Center the image
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image
    
    # Resize to target
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def masked_grain_square_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_size: int = 224,
) -> Optional[np.ndarray]:
    """Crop one grain, remove pixels outside its instance mask, and pad consistently."""
    if mask.ndim != 2 or image_rgb.shape[:2] != mask.shape:
        raise ValueError("Grain mask must be 2D and match the image dimensions")
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return None

    x, y, width, height = cv2.boundingRect(coords)
    crop_rgb = image_rgb[y:y + height, x:x + width].copy()
    crop_mask = mask[y:y + height, x:x + width] > 0
    crop_rgb[~crop_mask] = 0
    return pad_to_square(crop_rgb, target_size=target_size)


def image_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB numpy array to BGR for OpenCV operations."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR numpy array to RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
