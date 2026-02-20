#!/usr/bin/env python3
"""
CircuitSeeker multimodal registration (cluster-friendly, Otsu masking).

Pipeline:
- load TIFFs (Z,Y,X) -> (X,Y,Z)
- optional Z padding (µm)
- masks: normalize -> gaussian -> Otsu -> largest CC -> fill -> closing -> dilation
- principal axes (masks) -> modes
- alignment_pipeline: rigid -> affine -> deform
- save: masks, modes, affine, deform field, aligned volumes, metadata, log

Notes:
- Spacing arguments are in microns, order X Y Z (NO commas).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import nrrd
import tifffile as tiff
import SimpleITK as sitk
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing, binary_dilation
from skimage.filters import threshold_otsu
from skimage.measure import label

from CircuitSeeker.axisalign import principal_axes, align_modes
from CircuitSeeker.align import alignment_pipeline
from CircuitSeeker.transform import apply_transform


# -----------------------------
# Utilities
# -----------------------------

def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def setup_logging(log_path: Path, verbose: bool = False) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("cs_reg")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def to_xyz(vol_zyx: np.ndarray) -> np.ndarray:
    """Convert TIFF order (Z,Y,X) -> (X,Y,Z)."""
    if vol_zyx.ndim != 3:
        raise ValueError(f"Expected 3D TIFF stack (Z,Y,X). Got {vol_zyx.shape}")
    return np.transpose(vol_zyx, (2, 1, 0))

def to_zyx(vol_xyz: np.ndarray) -> np.ndarray:
    """Convert (X,Y,Z) -> TIFF order (Z,Y,X)."""
    if vol_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume (X,Y,Z). Got {vol_xyz.shape}")
    return np.transpose(vol_xyz, (2, 1, 0))

def pad_z_um(vol_xyz: np.ndarray, spacing_xyz: np.ndarray, pad_um: float, pad_value: float = 0.0) -> Tuple[np.ndarray, int]:
    """Pad in Z by pad_um on both ends (physical units)."""
    if pad_um <= 0:
        return vol_xyz, 0
    pad_vox = int(np.round(pad_um / float(spacing_xyz[2])))
    if pad_vox <= 0:
        return vol_xyz, 0
    pad_width = ((0, 0), (0, 0), (pad_vox, pad_vox))
    return np.pad(vol_xyz, pad_width=pad_width, mode="constant", constant_values=pad_value), pad_vox

def save_path(out_dir: Path, prefix: str, stem: str, ext: str) -> Path:
    return out_dir / f"{prefix}_{stem}.{ext}"


def otsu_fill_mask(
    vol_xyz: np.ndarray,
    sigma: float = 2.0,
    close_shape: Tuple[int, int, int] = (5, 5, 5),
    dilate_iter: int = 64,
) -> np.ndarray:
    """
    normalize -> gaussian -> Otsu -> largest CC -> fill -> closing -> dilation
    returns uint8 {0,1}
    """
    v = vol_xyz.astype(np.float32, copy=False)
    p1, p99 = np.percentile(v, (1, 99))
    v = np.clip((v - p1) / (p99 - p1 + 1e-6), 0, 1)

    v = gaussian_filter(v, sigma=float(sigma))

    t = threshold_otsu(v)
    m = v > t

    lbl = label(m)
    if lbl.max() > 0:
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        m = lbl == counts.argmax()
    else:
        # if empty, return empty mask
        return np.zeros_like(vol_xyz, dtype=np.uint8)

    m = binary_fill_holes(m)
    m = binary_closing(m, structure=np.ones(close_shape, dtype=bool))
    m = binary_dilation(m, iterations=int(dilate_iter))

    return m.astype(np.uint8)


# -----------------------------
# ImageJ overlay helpers
# -----------------------------

def _to_uint16_for_imagej(vol_zyx: np.ndarray) -> np.ndarray:
    """Convert a ZYX volume to uint16 for ImageJ overlays (robust display scaling)."""
    if np.issubdtype(vol_zyx.dtype, np.floating):
        p1, p99 = np.percentile(vol_zyx, (0.1, 99.9))
        if p99 > p1:
            v = (vol_zyx - p1) / (p99 - p1)
        else:
            v = np.clip(vol_zyx, 0, 1)
        return (np.clip(v, 0, 1) * 65535).astype(np.uint16)
    return np.clip(vol_zyx, 0, 65535).astype(np.uint16)


def write_imagej_overlay_tiff(
    fixed_tif: Path,
    warped_tif: Path,
    overlay_out: Path,
) -> None:
    """Write an ImageJ 2-channel overlay TIFF with axes TZCYX."""
    fixed_img = sitk.ReadImage(str(fixed_tif), sitk.sitkFloat32)
    if fixed_img.GetNumberOfComponentsPerPixel() > 1:
        fixed_img = sitk.VectorIndexSelectionCast(fixed_img, 0)
    fixed_arr = sitk.GetArrayFromImage(fixed_img)  # Z,Y,X

    warped_img = sitk.ReadImage(str(warped_tif), sitk.sitkFloat32)
    if warped_img.GetNumberOfComponentsPerPixel() > 1:
        warped_img = sitk.VectorIndexSelectionCast(warped_img, 0)
    warped_arr = sitk.GetArrayFromImage(warped_img)  # Z,Y,X

    fixed_u16 = _to_uint16_for_imagej(fixed_arr)
    warped_u16 = _to_uint16_for_imagej(warped_arr)

    zmin = min(fixed_u16.shape[0], warped_u16.shape[0])
    stacked = np.stack([fixed_u16[:zmin], warped_u16[:zmin]], axis=1)  # (Z,C,Y,X)
    stacked = stacked[np.newaxis, ...]  # (T,Z,C,Y,X)

    tiff.imwrite(str(overlay_out), stacked, bigtiff=True, imagej=True, metadata={"axes": "TZCYX"})


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class RunConfig:
    fixed: Path
    moving: Path
    fixed_spacing_um: np.ndarray
    moving_spacing_um: np.ndarray

    out_dir: Path
    exp_id: str
    fish: int
    run_tag: str

    pad_um: float = 20.0

    # Mask parameters (homogeneous)
    mask_sigma: float = 2.0
    mask_close_shape: Tuple[int, int, int] = (5, 5, 5)
    mask_dilate_iter: int = 64

    # Alignment parameters
    alignment_spacing: float = 2.0
    shrink_factors: Sequence[int] = (2,)
    smooth_sigmas: Sequence[float] = (8.0,)
    iterations: int = 400
    control_point_spacing: float = 10.0
    control_point_levels: Sequence[int] = (1, 2, 4, 8, 16, 32, 64)

    # I/O
    save_padded_tiffs: bool = True

    # Reproducibility
    seed: Optional[int] = None

    @property
    def run_id(self) -> str:
        return f"{self.exp_id}_fish{self.fish}"

    @property
    def prefix(self) -> str:
        # Used for ALL output names
        return self.run_tag


def write_meta(cfg: RunConfig, meta_path: Path) -> None:
    meta = {
        "exp_id": cfg.exp_id,
        "fish": int(cfg.fish),
        "run_id": cfg.run_id,
        "run_tag": cfg.run_tag,
        "fixed": str(cfg.fixed),
        "moving": str(cfg.moving),
        "fixed_spacing_um_xyz": [float(x) for x in cfg.fixed_spacing_um],
        "moving_spacing_um_xyz": [float(x) for x in cfg.moving_spacing_um],
        "pad_um": float(cfg.pad_um),
        "mask_sigma": float(cfg.mask_sigma),
        "mask_close_shape": list(cfg.mask_close_shape),
        "mask_dilate_iter": int(cfg.mask_dilate_iter),
        "alignment_spacing": float(cfg.alignment_spacing),
        "shrink_factors": list(cfg.shrink_factors),
        "smooth_sigmas": list(cfg.smooth_sigmas),
        "iterations": int(cfg.iterations),
        "control_point_spacing": float(cfg.control_point_spacing),
        "control_point_levels": list(cfg.control_point_levels),
        "save_padded_tiffs": bool(cfg.save_padded_tiffs),
        "seed": cfg.seed,
        "env": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME"),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -----------------------------
# Main pipeline
# -----------------------------


def run(cfg: RunConfig, logger: logging.Logger) -> None:
    ensure_dir(cfg.out_dir)

    meta_path = save_path(cfg.out_dir, cfg.prefix, "meta", "json")
    write_meta(cfg, meta_path)
    logger.info("Wrote meta: %s", meta_path)

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        logger.info("Set numpy random seed to %d", cfg.seed)

    logger.info("Loading fixed TIFF:  %s", cfg.fixed)
    logger.info("Loading moving TIFF: %s", cfg.moving)

    fixed_zyx = imread(str(cfg.fixed))
    moving_zyx = imread(str(cfg.moving))

    fix = to_xyz(fixed_zyx).astype(np.float32, copy=False)
    mov = to_xyz(moving_zyx).astype(np.float32, copy=False)

    fix_spacing = np.array(cfg.fixed_spacing_um, dtype=float)
    mov_spacing = np.array(cfg.moving_spacing_um, dtype=float)

    logger.info("Fixed shape XYZ:  %s", fix.shape)
    logger.info("Moving shape XYZ: %s", mov.shape)
    logger.info("Fixed spacing XYZ (um):  %s", fix_spacing)
    logger.info("Moving spacing XYZ (um): %s", mov_spacing)

    # Pad Z
    fixed_padded_tif_for_overlay = None
    if cfg.pad_um > 0:
        logger.info("Padding Z by ±%.3f um", cfg.pad_um)
        fix, fix_pad = pad_z_um(fix, fix_spacing, cfg.pad_um, pad_value=float(np.min(fix)))
        mov, mov_pad = pad_z_um(mov, mov_spacing, cfg.pad_um, pad_value=float(np.min(mov)))
        logger.info("After pad, fixed shape XYZ:  %s (pad_vox=%d)", fix.shape, fix_pad)
        logger.info("After pad, moving shape XYZ: %s (pad_vox=%d)", mov.shape, mov_pad)

        if cfg.save_padded_tiffs:
            fix_padded_path = save_path(cfg.out_dir, cfg.prefix, f"fixed_pad{int(cfg.pad_um)}um", "tif")
            mov_padded_path = save_path(cfg.out_dir, cfg.prefix, f"moving_pad{int(cfg.pad_um)}um", "tif")
            imwrite(fix_padded_path, to_zyx(fix), bigtiff=True)
            imwrite(mov_padded_path, to_zyx(mov), bigtiff=True)
            logger.info("Saved padded TIFFs: %s , %s", fix_padded_path, mov_padded_path)
            fixed_padded_tif_for_overlay = fix_padded_path

    # Masks (homogeneous)
    logger.info("Computing Otsu masks (sigma=%.2f, close=%s, dilate_iter=%d)",
                cfg.mask_sigma, cfg.mask_close_shape, cfg.mask_dilate_iter)

    fix_mask = otsu_fill_mask(
        fix,
        sigma=cfg.mask_sigma,
        close_shape=cfg.mask_close_shape,
        dilate_iter=cfg.mask_dilate_iter,
    )
    mov_mask = otsu_fill_mask(
        mov,
        sigma=cfg.mask_sigma,
        close_shape=cfg.mask_close_shape,
        dilate_iter=cfg.mask_dilate_iter,
    )

    if not np.any(fix_mask):
        raise RuntimeError("Fixed mask is empty (Otsu). Try changing sigma/dilate_iter or invert intensities.")
    if not np.any(mov_mask):
        raise RuntimeError("Moving mask is empty (Otsu). Try changing sigma/dilate_iter or invert intensities.")

    nrrd.write(str(save_path(cfg.out_dir, cfg.prefix, "fixed_mask", "nrrd")), fix_mask, compression_level=2)
    nrrd.write(str(save_path(cfg.out_dir, cfg.prefix, "moving_mask", "nrrd")), mov_mask, compression_level=2)
    logger.info("Saved masks.")

    # Principal axes -> modes
    logger.info("Principal-axes alignment")
    fix_mean, fix_evals, fix_evecs = principal_axes(fix_mask, fix_spacing)
    mov_mean, mov_evals, mov_evecs = principal_axes(mov_mask, mov_spacing)

    modes = align_modes(fix_mean, fix_evecs, mov_mean, mov_evecs)
    np.savetxt(save_path(cfg.out_dir, cfg.prefix, "modes", "mat"), modes)

    modes_aligned = apply_transform(
        fix, mov,
        fix_spacing, mov_spacing,
        transform_list=[modes],
    )
    nrrd.write(str(save_path(cfg.out_dir, cfg.prefix, "modes_aligned", "nrrd")), modes_aligned, compression_level=2)
    logger.info("Saved modes + modes_aligned.")

    # Global alignment
    logger.info("Global alignment: rigid → affine → deform")
    affine, deform = alignment_pipeline(
        fix, mov, fix_spacing, mov_spacing,
        steps=["rigid", "affine", "deform"],
        initial_transform=modes,
        alignment_spacing=float(cfg.alignment_spacing),
        shrink_factors=list(cfg.shrink_factors),
        smooth_sigmas=list(cfg.smooth_sigmas),
        iterations=int(cfg.iterations),
        deform_kwargs={
            "control_point_spacing": float(cfg.control_point_spacing),
            "control_point_levels": list(cfg.control_point_levels),
        },
    )
    deform_field = deform[1]

    affine_aligned = apply_transform(fix, mov, fix_spacing, mov_spacing, transform_list=[affine])
    deform_aligned = apply_transform(fix, mov, fix_spacing, mov_spacing, transform_list=[affine, deform_field])

    np.savetxt(save_path(cfg.out_dir, cfg.prefix, "affine", "mat"), affine)
    nrrd.write(str(save_path(cfg.out_dir, cfg.prefix, "deform", "nrrd")), deform_field, compression_level=2)
    nrrd.write(str(save_path(cfg.out_dir, cfg.prefix, "affine_aligned", "nrrd")), affine_aligned, compression_level=2)
    nrrd.write(str(save_path(cfg.out_dir, cfg.prefix, "deform_aligned", "nrrd")), deform_aligned, compression_level=2)

    # Save affine-aligned as TIFF (for ImageJ overlay)
    affine_aligned_tif = save_path(cfg.out_dir, cfg.prefix, "affine_aligned", "tif")
    imwrite(affine_aligned_tif, to_zyx(affine_aligned), bigtiff=True)

    # Build ImageJ overlay between padded fixed stack and affine-aligned output
    if cfg.save_padded_tiffs:
        overlay_out = save_path(cfg.out_dir, cfg.prefix, "overlay_fixedpad_vs_affine", "tif")
        logger.info(">> Building ImageJ 2-channel overlay (fixed padded, affine-aligned): %s", overlay_out
        )
        write_imagej_overlay_tiff(
            fixed_tif=fixed_padded_tif_for_overlay,
            warped_tif=affine_aligned_tif,
            overlay_out=overlay_out,
        )
        logger.info("Saved overlay TIFF: %s", overlay_out)
    else:
        logger.info("Skipping overlay: --save-padded-tiffs not set.")

    logger.info("Done. Outputs under: %s", cfg.out_dir)


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CircuitSeeker registration (Otsu masking, cluster-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--fixed", required=True, help="Fixed image TIFF (Z,Y,X).")
    p.add_argument("--moving", required=True, help="Moving image TIFF (Z,Y,X).")
    p.add_argument("--fixed-spacing-um", required=True, nargs=3, type=float, metavar=("SX", "SY", "SZ"),
                   help="Fixed spacing in microns, order X Y Z (NO commas).")
    p.add_argument("--moving-spacing-um", required=True, nargs=3, type=float, metavar=("SX", "SY", "SZ"),
                   help="Moving spacing in microns, order X Y Z (NO commas).")

    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--exp-id", required=True, help="Experiment identifier.")
    p.add_argument("--fish", required=True, type=int, help="Fish integer.")

    p.add_argument("--pad-um", type=float, default=20.0, help="Pad ±Z in microns before registration.")
    p.add_argument("--save-padded-tiffs", action="store_true", help="Write padded TIFFs to output directory.")

    # Mask params
    p.add_argument("--mask-sigma", type=float, default=2.0, help="Gaussian sigma for Otsu masking.")
    p.add_argument("--mask-dilate-iter", type=int, default=64, help="Binary dilation iterations (mask margin).")
    p.add_argument("--mask-close-shape", nargs=3, type=int, default=(5, 5, 5),
                   metavar=("CX", "CY", "CZ"), help="Binary closing struct size (odd ints recommended).")

    # Alignment params
    p.add_argument("--alignment-spacing", type=float, default=2.0, help="Skip-sample target spacing (µm).")
    p.add_argument("--shrink-factors", nargs="+", type=int, default=[2], help="Multi-res shrink factors.")
    p.add_argument("--smooth-sigmas", nargs="+", type=float, default=[8.0], help="Multi-res smoothing sigmas.")
    p.add_argument("--iterations", type=int, default=400, help="Registration iterations (passed to CircuitSeeker).")
    p.add_argument("--control-point-spacing", type=float, default=10.0, help="BSpline control point spacing (µm).")
    p.add_argument("--control-point-levels", nargs="+", type=int, default=[1, 2, 4, 8],
                   help="Scale factors for BSpline levels (coarse->fine).")

    p.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging to stdout.")
    p.add_argument("--log-file", default=None, help="Optional log file path.")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = _as_path(args.out_dir)
    ensure_dir(out_dir)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.exp_id}_fish{args.fish}_{ts}"
    run_out_dir = out_dir / run_tag
    ensure_dir(run_out_dir)

    cfg = RunConfig(
        fixed=_as_path(args.fixed),
        moving=_as_path(args.moving),
        fixed_spacing_um=np.array(args.fixed_spacing_um, dtype=float),
        moving_spacing_um=np.array(args.moving_spacing_um, dtype=float),
        out_dir=run_out_dir,
        exp_id=str(args.exp_id),
        fish=int(args.fish),
        run_tag=str(run_tag),
        pad_um=float(args.pad_um),
        mask_sigma=float(args.mask_sigma),
        mask_close_shape=tuple(int(x) for x in args.mask_close_shape),
        mask_dilate_iter=int(args.mask_dilate_iter),
        alignment_spacing=float(args.alignment_spacing),
        shrink_factors=tuple(int(x) for x in args.shrink_factors),
        smooth_sigmas=tuple(float(x) for x in args.smooth_sigmas),
        iterations=int(args.iterations),
        control_point_spacing=float(args.control_point_spacing),
        control_point_levels=tuple(int(x) for x in args.control_point_levels),
        save_padded_tiffs=bool(args.save_padded_tiffs),
        seed=args.seed,
    )

    # Logging
    default_log = cfg.out_dir / f"{cfg.prefix}_run.log"
    log_path = _as_path(args.log_file) if args.log_file else default_log
    logger = setup_logging(log_path, verbose=bool(args.verbose))

    # Sanity
    if not cfg.fixed.exists():
        logger.error("Fixed TIFF not found: %s", cfg.fixed)
        return 2
    if not cfg.moving.exists():
        logger.error("Moving TIFF not found: %s", cfg.moving)
        return 2

    logger.info("Run ID: %s", cfg.run_id)
    logger.info("python: %s", sys.executable)
    logger.info("CircuitSeeker: %s", __import__("CircuitSeeker").__file__)
    logger.info("SLURM_CPUS_PER_TASK: %s", os.environ.get("SLURM_CPUS_PER_TASK"))
    logger.info("OMP_NUM_THREADS: %s", os.environ.get("OMP_NUM_THREADS"))
    logger.info("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS: %s", os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"))

    try:
        run(cfg, logger)
    except Exception as e:
        logger.exception("Run failed with error: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())