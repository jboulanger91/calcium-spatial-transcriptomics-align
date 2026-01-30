#!/usr/bin/env python3
"""
ants_affine_minimal.py

Minimal ANTs affine registration for 3D volumes, with optional masks computed
from TIFF inputs (using CircuitSeeker level_set), and conversion to NIfTI for ANTs.

Inputs:
  --fixed   fixed reference volume (.tif/.tiff)
  --moving  moving volume (.tif/.tiff)
  --fixed-spacing-um  X Y Z   (microns)
  --moving-spacing-um X Y Z   (microns)
  --out-dir output directory
  --exp-id  e.g. exp_001
  --fish    e.g. 2
  --use-masks        compute masks from TIFFs and pass to antsRegistration
  --mask-downsample  downsample factor for masking (default 4)

Outputs (in out-dir):
  exp_001_fish2_fixed.nii.gz
  exp_001_fish2_moving.nii.gz
  exp_001_fish2_fix_mask.nrrd       (if --use-masks)
  exp_001_fish2_mov_mask.nrrd       (if --use-masks)
  exp_001_fish2_warped.nii.gz
  exp_001_fish2_affine.mat

Example usage:
  python3 ants_affine_minimal.py \
  --fixed  "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/fish2/prealigned/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_ch1.tif" \
  --moving "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/2p_stacks/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz.tif" \
  --fixed-spacing-um 1 1 1.0 \
  --moving-spacing-um 1 1 2.0\
  --out-dir "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/ANTs_output" \
  --exp-id exp_001 \
  --fish 2 \
  --use-masks \
  --mask-downsample 2
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import nrrd
from tifffile import imread
from scipy.ndimage import zoom, binary_closing, binary_dilation
from CircuitSeeker import level_set


# Ensure ANTs binaries are in PATH (edit if needed)
os.environ["PATH"] = "/Users/jonathanboulanger-weill/Packages/install/bin:" + os.environ["PATH"]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_tiff_xyz(tif_path: Path) -> np.ndarray:
    """Read a 3D TIFF and return a numpy volume in (X, Y, Z) order."""
    vol_zyx = imread(str(tif_path))
    if vol_zyx.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF (Z,Y,X). Got shape={vol_zyx.shape} for {tif_path}")
    return vol_zyx.transpose(2, 1, 0)  # -> (X,Y,Z)


def save_nii_from_xyz(vol_xyz: np.ndarray, spacing_xyz_um: np.ndarray, out_nii_gz: Path) -> None:
    """Save (X,Y,Z) numpy volume as NIfTI (.nii.gz) with spacing set in microns."""
    vol_zyx = vol_xyz.transpose(2, 1, 0)  # back to (Z,Y,X) for SimpleITK
    img = sitk.GetImageFromArray(vol_zyx)
    # SimpleITK spacing is (X,Y,Z)
    img.SetSpacing(tuple(float(x) for x in spacing_xyz_um))
    sitk.WriteImage(img, str(out_nii_gz), useCompression=True)


def save_mask_nii_from_xyz(mask_xyz: np.ndarray, spacing_xyz_um: np.ndarray, out_nii_gz: Path) -> None:
    """Save a binary mask (X,Y,Z) as NIfTI (.nii.gz) with correct spacing."""
    mask_zyx = mask_xyz.transpose(2, 1, 0)  # (Z,Y,X)
    img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    img.SetSpacing(tuple(float(x) for x in spacing_xyz_um))
    sitk.WriteImage(img, str(out_nii_gz), useCompression=True)


def brain_mask(vol_xyz: np.ndarray, spacing_xyz: np.ndarray, lambda2: float, ds: int = 4) -> np.ndarray:
    """Coarse brain mask via level_set on a downsampled volume, then upsample + smooth."""
    ds = max(int(ds), 1)
    vol_skip = vol_xyz[::ds, ::ds, ::ds]
    skip_spacing = spacing_xyz * np.array([ds, ds, ds], dtype=float)

    t0 = time.time()
    print(f"    [mask] downsample={ds} vol_skip={vol_skip.shape} spacing_um={skip_spacing}")

    mask_small = level_set.brain_detection(
        vol_skip,
        skip_spacing,
        mask_smoothing=2,
        iterations=[80, 40, 10],
        smooth_sigmas=[12, 6, 3],
        lambda2=lambda2,
    )

    print(f"    [mask] level_set done in {time.time() - t0:.1f}s")

    mask = zoom(mask_small, np.array(vol_xyz.shape) / np.array(vol_skip.shape), order=0)
    mask = binary_closing(mask, np.ones((5, 5, 5))).astype(np.uint8)
    mask = binary_dilation(mask, np.ones((5, 5, 5))).astype(np.uint8)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True, help="Fixed volume (.tif/.tiff)")
    ap.add_argument("--moving", required=True, help="Moving volume (.tif/.tiff)")
    ap.add_argument("--fixed-spacing-um", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--moving-spacing-um", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--out-dir", required=True, help="Output folder")
    ap.add_argument("--exp-id", required=True, help="e.g. exp_001")
    ap.add_argument("--fish", type=int, required=True, help="e.g. 2")
    ap.add_argument("--use-masks", action="store_true", help="Compute masks from TIFFs and pass to ANTs")
    ap.add_argument("--mask-downsample", type=int, default=4, help="Downsample factor for mask computation")
    args = ap.parse_args()

    fixed_tif = Path(args.fixed).resolve()
    moving_tif = Path(args.moving).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fix_spacing = np.array(args.fixed_spacing_um, dtype=float)
    mov_spacing = np.array(args.moving_spacing_um, dtype=float)

    print("[info] spacings (um):")
    print("  fixed :", fix_spacing)
    print("  moving:", mov_spacing)

    # -------------------------
    # Load TIFFs as numpy XYZ
    # -------------------------
    if fixed_tif.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("This version expects --fixed to be a .tif/.tiff (so we can compute masks).")
    if moving_tif.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("This version expects --moving to be a .tif/.tiff (so we can compute masks).")

    print(f"[load] fixed TIFF : {fixed_tif}")
    fix_xyz = read_tiff_xyz(fixed_tif)
    print(f"[load] moving TIFF: {moving_tif}")
    mov_xyz = read_tiff_xyz(moving_tif)

    print("[info] shapes (X,Y,Z):")
    print("  fixed :", fix_xyz.shape)
    print("  moving:", mov_xyz.shape)

    # -------------------------
    # Convert to NIfTI for ANTs
    # -------------------------
    fixed_nii = out_dir / f"{args.exp_id}_fish{args.fish}_fixed.nii.gz"
    moving_nii = out_dir / f"{args.exp_id}_fish{args.fish}_moving.nii.gz"

    print(f"[write] fixed NIfTI : {fixed_nii}")
    save_nii_from_xyz(fix_xyz, fix_spacing, fixed_nii)

    print(f"[write] moving NIfTI: {moving_nii}")
    save_nii_from_xyz(mov_xyz, mov_spacing, moving_nii)

    # -------------------------
    # Masks (optional)
    # -------------------------
    fix_mask_nrrd = out_dir / f"{args.exp_id}_fish{args.fish}_fix_mask.nrrd"
    mov_mask_nrrd = out_dir / f"{args.exp_id}_fish{args.fish}_mov_mask.nrrd"
    fix_mask_nii = out_dir / f"{args.exp_id}_fish{args.fish}_fix_mask.nii.gz"
    mov_mask_nii = out_dir / f"{args.exp_id}_fish{args.fish}_mov_mask.nii.gz"

    masks_arg = None
    if args.use_masks:
        print("[mask] computing fixed mask...")
        fix_mask = brain_mask(fix_xyz.astype(np.float32, copy=False), fix_spacing, lambda2=32.0, ds=args.mask_downsample)
        print("[mask] computing moving mask...")
        mov_mask = brain_mask(mov_xyz.astype(np.float32, copy=False), mov_spacing, lambda2=64.0, ds=args.mask_downsample)

        # Save NRRD (for inspection/debugging)
        nrrd.write(str(fix_mask_nrrd), fix_mask)
        nrrd.write(str(mov_mask_nrrd), mov_mask)

        # Save NIfTI (for ANTs)
        save_mask_nii_from_xyz(fix_mask, fix_spacing, fix_mask_nii)
        save_mask_nii_from_xyz(mov_mask, mov_spacing, mov_mask_nii)

        print("[save] masks:")
        print("  fix nrrd:", fix_mask_nrrd)
        print("  mov nrrd:", mov_mask_nrrd)
        print("  fix nii :", fix_mask_nii)
        print("  mov nii :", mov_mask_nii)

        masks_arg = f"[{fix_mask_nii},{mov_mask_nii}]"
        print(f"[ants] using masks: {masks_arg}")

    # -------------------------
    # ANTs registration
    # -------------------------
    # ANTs uses an output "prefix" for transform filenames (e.g. *0GenericAffine.mat).
    # For the warped image, we write directly to the canonical name to avoid
    # case-insensitive filesystem collisions (Warped vs warped).
    prefix = out_dir / f"{args.exp_id}_fish{args.fish}_"
    warped_dst = out_dir / f"{args.exp_id}_fish{args.fish}_warped.nii.gz"
    inverse_warped_dst = out_dir / f"{args.exp_id}_fish{args.fish}_inversewarped.nii.gz"
    affine_dst = out_dir / f"{args.exp_id}_fish{args.fish}_affine.mat"

    print("[ants] running antsRegistration...")

    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--output", f"[{prefix},{warped_dst},{inverse_warped_dst}]",
        "--write-composite-transform", "0",

        # Rigid
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x300,1e-8,10]",
        "--smoothing-sigmas", "3x2x1x0",
        "--shrink-factors", "8x4x2x1",

        # Affine
        "--transform", "Affine[0.1]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-8,10]",
        "--smoothing-sigmas", "3x2x1x0",
        "--shrink-factors", "8x4x2x1",

        # SyN (deformable)
        #"--transform", "SyN[0.1,6,0]",
        #"--metric", f"CC[{fixed_nii},{moving_nii},1,2]",
        #"--convergence", "[200x200x200x100,1e-7,10]",
        #"--smoothing-sigmas", "4x3x2x1",
        #"--shrink-factors", "12x8x4x2",
    ]

    if masks_arg is not None:
        cmd += ["--masks", masks_arg]

    run(cmd)
    print("[ants] done")

    # -------------------------
    # Canonical output naming
    # -------------------------
    # Warped output is written directly to `warped_dst` by antsRegistration.
    affine_src = Path(f"{prefix}0GenericAffine.mat")

    if not Path(warped_dst).exists():
        raise FileNotFoundError(f"Expected warped output not found: {warped_dst}")
    if not affine_src.exists():
        raise FileNotFoundError(f"Expected affine transform not found: {affine_src}")

    # Copy the affine to the canonical name (safe across case-insensitive filesystems)
    shutil.copy2(affine_src, affine_dst)

    print(">> Done. Wrote:")
    print("  ", warped_dst)
    print("  ", affine_dst)


if __name__ == "__main__":
    main()