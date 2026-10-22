"""Conservative input validation for the upload path.

This module intentionally sits in front of preprocessing/inference.  It does
not change the model input contract; it only rejects inputs that are unlikely
to be a single handwritten digit.
"""
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


INVALID_UPLOAD_MESSAGE = (
    "No clear handwritten digit detected.\n"
    "Upload or crop an image containing one handwritten digit from 0 to 9."
)
SUPPORTED_FORMATS = {"PNG", "JPEG"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    message: str = ""
    reason: str = ""


def _reject(reason: str) -> ValidationResult:
    return ValidationResult(False, INVALID_UPLOAD_MESSAGE, reason)


def _gray_and_alpha(image: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if image.width == 0 or image.height == 0:
            raise ValueError("empty image")
        mode = image.mode
        array = np.asarray(image)
        alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255 if "A" in mode else None
    else:
        array = np.asarray(image)
        alpha = None
    if array.size == 0:
        raise ValueError("empty image")
    if array.ndim == 2:
        gray = array.astype(np.float32)
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if array.shape[-1] == 4:
            alpha = array[..., 3].astype(np.float32) / 255
    else:
        raise ValueError("unsupported image shape")
    if gray.max(initial=0) > 1:
        gray /= 255.0
    if alpha is not None:
        # Composite RGBA inputs onto white so a transparent background does
        # not become an artificial black rectangle during suitability checks.
        gray = gray * alpha + (1.0 - alpha)
    return np.clip(gray, 0, 1), alpha


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return connected component coordinates using an 8-neighbour scan."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for y, x in zip(*np.where(mask & ~seen)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        points = []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        components.append(np.asarray(points, dtype=np.int32))
    return components


def validate_image(image: Any) -> ValidationResult:
    """Validate that *image* is suitable for single-digit prediction."""
    try:
        gray, alpha = _gray_and_alpha(image)
    except (OSError, ValueError, TypeError, UnidentifiedImageError):
        return _reject("unreadable or unsupported")

    if gray.ndim != 2 or min(gray.shape) < 8:
        return _reject("empty or too small")
    if alpha is not None and float(alpha.max(initial=0)) < 0.05:
        return _reject("transparent image")

    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    deviation = np.abs(gray - background)
    contrast = float(np.percentile(deviation, 99) - np.percentile(deviation, 50))
    if contrast < 0.055 or float(gray.std()) < 0.018:
        return _reject("blank or uniform")

    threshold = max(0.075, float(np.percentile(deviation, 75)) * 0.55, float(deviation.std()) * 1.5)
    mask = deviation > threshold
    # A one-pixel border is usually compression/noise, not handwriting.
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    min_area = max(3, int(gray.size * 0.0012))
    components = [component for component in _components(mask) if len(component) >= min_area]
    if not components:
        return _reject("no foreground")
    components.sort(key=len, reverse=True)
    largest = components[0]
    foreground = sum(len(component) for component in components)
    occupancy = foreground / gray.size
    ys, xs = largest[:, 0], largest[:, 1]
    bbox_height, bbox_width = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    aspect = bbox_width / bbox_height
    bbox_occupancy = (bbox_height * bbox_width) / gray.size

    if len(components) > 1 and len(components[1]) > max(min_area * 2, len(largest) * 0.22):
        return _reject("multiple objects")
    if not 0.08 <= aspect <= 4.5 or occupancy > 0.48 or bbox_occupancy > 0.92:
        return _reject("not a single digit")
    # A broad, dense silhouette filling a large square/rectangular crop is
    # characteristic of an object photograph or icon, not a digit stroke.
    if occupancy > 0.30 and bbox_occupancy > 0.35:
        return _reject("object-like photo")

    # Photos tend to contain colour and high-frequency texture.  The checks
    # are deliberately conservative and only reject clear photo-like inputs.
    if isinstance(image, Image.Image) and image.mode in ("RGB", "RGBA"):
        rgb = np.asarray(image)[..., :3].astype(np.float32) / 255
        saturation = float(np.mean(np.max(rgb, axis=-1) - np.min(rgb, axis=-1)))
        if saturation > 0.16:
            return _reject("natural photo")
    gradient = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    if gradient > 0.22 and occupancy > 0.12:
        return _reject("natural or object photo")
    return ValidationResult(True)


def validate_upload(upload: Any) -> ValidationResult:
    """Validate a Streamlit upload, including decode and format failures."""
    try:
        filename = str(getattr(upload, "name", "")).lower()
        if filename and Path(filename).suffix.lstrip(".").upper() not in {"PNG", "JPG", "JPEG"}:
            return _reject("unsupported file format")
        size = getattr(upload, "size", None)
        if size is None and hasattr(upload, "getbuffer"):
            size = len(upload.getbuffer())
        if size is not None and int(size) > MAX_UPLOAD_BYTES:
            return _reject("file is too large")
        if isinstance(upload, (bytes, bytearray)):
            upload = io.BytesIO(upload)
        if hasattr(upload, "seek"):
            upload.seek(0)
        with Image.open(upload) as opened:
            opened.verify()
        if hasattr(upload, "seek"):
            upload.seek(0)
        with Image.open(upload) as opened:
            if opened.format not in SUPPORTED_FORMATS:
                return _reject("unsupported file format")
            if not (8 <= opened.width <= MAX_IMAGE_DIMENSION and 8 <= opened.height <= MAX_IMAGE_DIMENSION):
                return _reject("invalid image dimensions")
            image = opened.copy()
        return validate_image(image)
    except (OSError, ValueError, TypeError, UnidentifiedImageError):
        return _reject("unreadable or unsupported")


# Short alias useful to callers that already have a decoded PIL image.
validate = validate_image
