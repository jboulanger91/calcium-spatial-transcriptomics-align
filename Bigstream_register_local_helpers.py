from __future__ import annotations

"""Local helper utilities for BigStream registration notebooks.

Conventions:
- Volumes are indexed as (Z, Y, X).
- Spacing vectors are (Z, Y, X) in microns.

Design:
- No global state: output paths are always passed explicitly.
- Functions are small, notebook-friendly building blocks.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing, binary_dilation
from skimage.measure import label

import tifffile
from tifffile import imwrite as tiff_imwrite


def mask_path(out_path: Path, run_id: str, tag: str) -> Path:
    """Return an NRRD path for storing a mask (XYZ ordering is handled by the caller)."""
    return out_path / f"{run_id}_{tag}.nrrd"


def homogenize_then_threshold_mask(
    vol: np.ndarray,
    thresh=0.20,
    sigma_bg=25,
    sigma_smooth=1.5,
    close_shape=(5, 5, 5),
    dilate_iter=32,
    keep_largest=True,
) -> tuple[np.ndarray, np.ndarray]:
    """Homogenize intensities and create a foreground mask.

    Returns
    -------
    mask : uint8 ndarray
        Binary mask in the same (Z, Y, X) shape as `vol`.
    v_corr : float32 ndarray
        Homogenized/contrast-normalized volume useful for QC visualization.
    """
    v = vol.astype(np.float32, copy=False)

    p1, p99 = np.percentile(v, (1, 99))
    v = np.clip((v - p1) / (p99 - p1 + 1e-6), 0, 1)

    bg = gaussian_filter(v, sigma=sigma_bg)
    v_corr = np.clip(v / (bg + 1e-6), 0, np.percentile(v, 99.9))

    p1c, p99c = np.percentile(v_corr, (1, 99))
    v_corr = np.clip((v_corr - p1c) / (p99c - p1c + 1e-6), 0, 1)

    v_corr = gaussian_filter(v_corr, sigma=sigma_smooth)
    m = v_corr > float(thresh)

    if keep_largest:
        lbl = label(m)
        if lbl.max() > 0:
            counts = np.bincount(lbl.ravel()); counts[0] = 0
            m = (lbl == counts.argmax())

    m = binary_fill_holes(m)
    m = binary_closing(m, structure=np.ones(close_shape, bool))
    m = binary_dilation(m, iterations=int(dilate_iter))

    return m.astype(np.uint8), v_corr


def _to_uint16_for_imagej(vol_zyx: np.ndarray) -> np.ndarray:
    """Convert a (Z, Y, X) volume to uint16 for ImageJ overlays (robust percentile scaling)."""
    if np.issubdtype(vol_zyx.dtype, np.floating):
        p1, p99 = np.percentile(vol_zyx, (0.1, 99.9))
        v = (vol_zyx - p1) / (p99 - p1) if p99 > p1 else np.clip(vol_zyx, 0, 1)
        return (np.clip(v, 0, 1) * 65535).astype(np.uint16)
    return np.clip(vol_zyx, 0, 65535).astype(np.uint16)


def to_zyx(vol: np.ndarray) -> np.ndarray:
    """Ensure a volume is in (Z, Y, X) order.

    This project uses (Z, Y, X) everywhere; this helper is an identity function but
    keeps call sites explicit.
    """
    return vol


def save_path(out_path: Path, run_id: str, timestamp: str, tag: str, ext: str) -> Path:
    """Build a timestamped output path.

    Format: <run_id>_<YYYYMMDD_HHMMSS>_<tag>.<ext>
    """
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path / f"{run_id}_{timestamp}_{tag}.{ext}"


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy arrays and tuples into JSON-safe (lists/ints/floats) structures."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    return obj


def write_json(out_path: Path, run_id: str, timestamp: str, tag: str, payload: dict) -> Path:
    """Write a JSON file next to other outputs, sanitizing numpy objects."""
    out_path.mkdir(parents=True, exist_ok=True)
    out = save_path(out_path, run_id, timestamp, tag, "json")
    with open(out, "w") as f:
        json.dump(sanitize_for_json(payload), f, indent=2)
    print(" saved:", out)
    return out


def load_tiff_float32(path: Path | str) -> np.ndarray:
    """Memory-map a TIFF and return a float32 numpy array (minimal copying)."""
    arr = np.asarray(tifffile.memmap(path))
    return arr.astype(np.float32, copy=False)
