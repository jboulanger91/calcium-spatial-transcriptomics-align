#!/usr/bin/env python3
"""
ANTs_register.py

Convert 3D TIFF image stacks to NIfTI and register them using ANTs (antsRegistration).

This script is intended for registering volumetric imaging stacks of the same modality
(e.g. GCaMP → GCaMP) using a multi-stage ANTs pipeline:
Rigid (MI) → Affine (MI) → SyN (CC).

Key requirements and assumptions
--------------------------------
1) Correct pixel/voxel resolution is critical.
   ANTs optimizes registration in *physical space* (millimeters), not pixel space.
   Therefore, voxel spacing must be correct for both fixed and moving images.
   This script explicitly sets spacing metadata during TIFF → NIfTI conversion
   (via --*-spacing-mm or --*-spacing-um). Incorrect spacing will lead to
   incorrect transforms even if the visual alignment appears reasonable.

2) Orientation and flipping must already be correct.
   This script does NOT handle rotations, flips, or axis reorientation.
   It assumes both stacks are already in the same anatomical orientation
   (e.g. rostro–caudal up, left/right consistent).
   Any required flipping or coarse reorientation should be done upstream
   (e.g. using a Napari-based pre-alignment step).

3) TIFF stacks are converted to NIfTI for robustness.
   While ANTs can sometimes read TIFF files directly, NIfTI (.nii/.nii.gz)
   provides reliable handling of:
     - voxel spacing
     - dimensionality (3D volumes)
     - physical coordinate metadata
   Converting TIFF → NIfTI ensures consistent behavior across systems and
   avoids silent metadata misinterpretation.

Inputs
------
- Fixed TIFF stack (reference volume)
- Moving TIFF stack (volume to be warped)
- Explicit voxel spacing for each stack (mm or µm)

Bash command example
--------------------
python3 ANTs_register.py \
  --fixed "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/fish2/prealigned/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_ch1.tif" \
  --moving "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/2p_stacks/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz.tif" \
  --fixed-spacing-um 0.621 0.621 1.0 \
  --moving-spacing-um 0.396 0.396 2.0\
  --out-prefix reg_ \
  --keep-nii

  python3 ANTs_register.py \
  --fixed "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/fish2/prealigned/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_ch1.tif" \
  --moving "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/2p_stacks/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz.tif" \
  --fixed-spacing-um 1 1 1.0 \
  --moving-spacing-um 1 1 2.0\
  --exp-id exp_001 \
  --fish 2 \
  --out-prefix reg_ \
  --keep-nii

Outputs
-------
- <out_prefix>Warped.nii.gz          : moving → fixed
- <out_prefix>InverseWarped.nii.gz   : fixed → moving
- <out_prefix>0GenericAffine.mat
- <out_prefix>1Warp.nii.gz / 1InverseWarp.nii.gz

No masking or cropping is applied at this stage; partial overlap is allowed.
Masking and overlap restriction can be added in later refinement steps.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk
import tifffile as tiff
import numpy as np

os.environ["PATH"] = "/Users/jonathanboulanger-weill/Packages/install/bin:" + os.environ["PATH"]

def short_name_from_fish(path: Path) -> str:
    """Return a shortened stem for filenames.

    If the substring 'fish' occurs in the stem, return the stem starting at 'fish'
    (to keep names short and consistent across pipelines). Otherwise return the full stem.
    """
    stem = path.stem
    idx = stem.find("fish")
    return stem[idx:] if idx != -1 else stem

def to_mm(sp):
    """Accept spacing as tuple in mm."""
    return tuple(float(x) for x in sp)

def um_to_mm(sp):
    """Convert microns → mm."""
    return tuple(float(x) / 1000.0 for x in sp)

def read_tif_write_nii(tif_path, nii_path, spacing_mm=None):
    # Force scalar float read; avoids some palette/colormap annoyances
    img = sitk.ReadImage(tif_path, sitk.sitkFloat32)

    # If image is multi-channel, take channel 0 (or use VectorMagnitude if you prefer)
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)

    if spacing_mm is not None:
        img.SetSpacing(tuple(map(float, spacing_mm)))  # ANTs expects mm

    sitk.WriteImage(img, nii_path)
    if not os.path.exists(nii_path):
        raise RuntimeError(f"Failed to write NIfTI: {nii_path}")
    return nii_path

def _to_uint16_for_imagej(vol: np.ndarray) -> np.ndarray:
    """Convert input volume to uint16 suitable for ImageJ display.

    - If float, rescale using 1st and 99th percentiles to [0, 65535].
    - If integer, clip to [0, 65535].
    """
    if np.issubdtype(vol.dtype, np.floating):
        p1, p99 = np.percentile(vol, (1, 99))
        if p99 > p1:
            vol_rescaled = (vol - p1) / (p99 - p1)
        else:
            vol_rescaled = np.clip(vol, 0, 1)
        vol_rescaled = np.clip(vol_rescaled, 0, 1)
        vol_uint16 = (vol_rescaled * 65535).astype(np.uint16)
    else:
        vol_uint16 = np.clip(vol, 0, 65535).astype(np.uint16)
    return vol_uint16

def write_fixed_warped_overlay_tiff(fixed_nii: Path, warped_nii: Path, out_tif: Path) -> None:
    """Write a 2-channel ImageJ-compatible TIFF with fixed and warped volumes.

    The output shape is (T=1, Z, C=2, Y, X).
    Channel 0 = fixed, Channel 1 = warped moving.
    """
    fixed_img = sitk.ReadImage(str(fixed_nii))
    warped_img = sitk.ReadImage(str(warped_nii))

    fixed_arr = sitk.GetArrayFromImage(fixed_img)  # Z,Y,X
    warped_arr = sitk.GetArrayFromImage(warped_img)  # Z,Y,X

    fixed_uint16 = _to_uint16_for_imagej(fixed_arr)
    warped_uint16 = _to_uint16_for_imagej(warped_arr)

    # Stack channels: shape (Z, C=2, Y, X)
    stacked = np.stack([fixed_uint16, warped_uint16], axis=1)

    # Add time dimension: shape (T=1, Z, C=2, Y, X)
    stacked = stacked[np.newaxis, ...]

    # Write TIFF with ImageJ metadata
    tiff.imwrite(str(out_tif), stacked, bigtiff=True, imagej=True, metadata={"axes": "TZCYX"})

# --- in run_ants_registration: remove the unsupported flag ---
def run_ants_registration(fixed_nii, moving_nii, out_prefix, shrink_factors: str, smoothing_sigmas: str,
                          rigid_step: float, affine_step: float, sampling: float, bins: int):
    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz,{out_prefix}InverseWarped.nii.gz]",
        "--write-composite-transform", "1",

        # Rigid (add finest level by including x1 shrink and x0 smoothing)
        "--transform", f"Rigid[{rigid_step}]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,{bins},Regular,{sampling}]",
        "--convergence", "[2000x1000x500x250,1e-8,10]",
        "--smoothing-sigmas", smoothing_sigmas,
        "--shrink-factors", shrink_factors,

        # Affine (finer step + same multires)
        "--transform", f"Affine[{affine_step}]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,{bins},Regular,{sampling}]",
        "--convergence", "[2000x1000x500x250,1e-8,10]",
        "--smoothing-sigmas", smoothing_sigmas,
        "--shrink-factors", shrink_factors,

        # SyN (optional)
        # "--transform", "SyN[0.1,6,0]",
        # "--metric", f"CC[{fixed_nii},{moving_nii},1,2]",
        # "--convergence", "[200x200x200x100,1e-7,10]",
        # "--smoothing-sigmas", "4x3x2x1",
        # "--shrink-factors", "12x8x4x2",
    ]
    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser(description="Convert two TIFF stacks to NIfTI with independent spacings and register with ANTs.")
    p.add_argument("--fixed", required=True, help="Fixed/reference TIFF stack")
    p.add_argument("--moving", required=True, help="Moving TIFF stack")

    # Global identifiers used for ALL output names (match CircuitSeeker naming)
    p.add_argument("--exp-id", required=True, help="Experiment identifier, e.g. 'exp_001' or 'exp1_110425'")
    p.add_argument("--fish", type=int, required=True, help="Fish integer identifier, e.g. 2")

    # Optional extra prefix segment (kept for backwards compatibility, deprecated)
    p.add_argument("--out-prefix", default="reg", help="(Deprecated) Extra prefix segment (default: 'reg'). Ignored for output naming.")

    # Output directory for ANTs results
    p.add_argument("--out-dir", default=(
        "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
        "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/"
        "data/exp1_110425/ANTs_output"
    ), help="Output directory for ANTs results and intermediates (default: hardcoded path)")

    # Spacing in mm
    p.add_argument("--fixed-spacing-mm",  type=float, nargs=3, metavar=("SX","SY","SZ"),
                   help="Fixed spacing in mm, e.g., 0.00048 0.00048 0.002")
    p.add_argument("--moving-spacing-mm", type=float, nargs=3, metavar=("SX","SY","SZ"),
                   help="Moving spacing in mm")

    # Or spacing in microns (um). If both are given, *_mm wins.
    p.add_argument("--fixed-spacing-um",  type=float, nargs=3, metavar=("SX","SY","SZ"),
                   help="Fixed spacing in microns (µm), e.g., 0.48 0.48 2.0")
    p.add_argument("--moving-spacing-um", type=float, nargs=3, metavar=("SX","SY","SZ"),
                   help="Moving spacing in microns (µm)")

    # Registration multi-resolution controls (to enable finer alignment)
    p.add_argument("--shrink-factors", default="8x4x2x1",
                   help="Multi-resolution shrink factors, e.g. '8x4x2x1' (must match smoothing sigmas levels)")
    p.add_argument("--smoothing-sigmas", default="4x2x1x0vox",
                   help="Multi-resolution smoothing sigmas, e.g. '4x2x1x0vox' (use '0vox' for finest level)")
    p.add_argument("--rigid-step", type=float, default=0.1,
                   help="Rigid stage step size, smaller = finer (default 0.1)")
    p.add_argument("--affine-step", type=float, default=0.05,
                   help="Affine stage step size, smaller = finer (default 0.05)")
    p.add_argument("--sampling", type=float, default=0.5,
                   help="Metric sampling fraction in [0,1], higher = more precise but slower (default 0.5)")
    p.add_argument("--mi-bins", type=int, default=64,
                   help="Mattes MI histogram bins (default 64)")

    p.add_argument("--keep-nii", action="store_true", help="Keep intermediate NIfTI files")
    args = p.parse_args()

    if shutil.which("antsRegistration") is None:
        sys.exit("Error: antsRegistration not found in PATH.")

    # Create ANTs output directory
    ants_dir = Path(args.out_dir).resolve()
    ants_dir.mkdir(parents=True, exist_ok=True)

    fixed_path = Path(args.fixed).resolve()
    moving_path = Path(args.moving).resolve()

    # Standardized output prefix (minimal, CircuitSeeker-compatible)
    # antsRegistration will write files like: <prefix>Warped.nii.gz, <prefix>0GenericAffine.mat, etc.
    prefix = ants_dir / f"{args.exp_id}_fish{args.fish}_"

    # Resolve spacings
    if args.fixed_spacing_mm:
        fixed_spacing_mm = to_mm(args.fixed_spacing_mm)
    elif args.fixed_spacing_um:
        fixed_spacing_mm = um_to_mm(args.fixed_spacing_um)
    else:
        fixed_spacing_mm = None  # use TIFF metadata if present

    if args.moving_spacing_mm:
        moving_spacing_mm = to_mm(args.moving_spacing_mm)
    elif args.moving_spacing_um:
        moving_spacing_mm = um_to_mm(args.moving_spacing_um)
    else:
        moving_spacing_mm = None

    tmpdir = tempfile.mkdtemp(prefix="tif2nii_")
    fixed_nii  = os.path.join(tmpdir, "fixed.nii.gz")
    moving_nii = os.path.join(tmpdir, "moving.nii.gz")

    try:
        print(">> Converting TIFF → NIfTI with independent spacings...")
        read_tif_write_nii(args.fixed, fixed_nii,  spacing_mm=fixed_spacing_mm)
        read_tif_write_nii(args.moving, moving_nii, spacing_mm=moving_spacing_mm)

        # Note: ANTs operates in physical space; different spacings are fine as long as they are correct.
        print(">> Running ANTs registration...")
        run_ants_registration(
            fixed_nii,
            moving_nii,
            str(prefix),
            shrink_factors=args.shrink_factors,
            smoothing_sigmas=args.smoothing_sigmas,
            rigid_step=args.rigid_step,
            affine_step=args.affine_step,
            sampling=args.sampling,
            bins=args.mi_bins,
        )

        # Standardize final filenames (rename/move ANTs outputs)
        # Desired example: exp_001_fish2_warped.nii.gz
        warped_src = Path(f"{prefix}Warped.nii.gz")
        invwarped_src = Path(f"{prefix}InverseWarped.nii.gz")
        affine_src = Path(f"{prefix}0GenericAffine.mat")

        warped_dst = ants_dir / f"{args.exp_id}_fish{args.fish}_warped.nii.gz"
        invwarped_dst = ants_dir / f"{args.exp_id}_fish{args.fish}_inverse_warped.nii.gz"
        affine_dst = ants_dir / f"{args.exp_id}_fish{args.fish}_affine.mat"

        if not warped_src.exists():
            raise FileNotFoundError(f"Expected warped output not found: {warped_src}")
        if not invwarped_src.exists():
            raise FileNotFoundError(f"Expected inverse-warped output not found: {invwarped_src}")
        if not affine_src.exists():
            raise FileNotFoundError(f"Expected affine transform not found: {affine_src}")

        # Overwrite if they already exist
        for dst in (warped_dst, invwarped_dst, affine_dst):
            if dst.exists():
                dst.unlink()

        shutil.move(str(warped_src), str(warped_dst))
        shutil.move(str(invwarped_src), str(invwarped_dst))
        shutil.move(str(affine_src), str(affine_dst))

        # Optional SyN warp files (only present if SyN stage is enabled)
        warp_src = Path(f"{prefix}1Warp.nii.gz")
        invwarp_src = Path(f"{prefix}1InverseWarp.nii.gz")
        warp_dst = ants_dir / f"{args.exp_id}_fish{args.fish}_warp.nii.gz"
        invwarp_dst = ants_dir / f"{args.exp_id}_fish{args.fish}_inverse_warp.nii.gz"
        if warp_src.exists():
            if warp_dst.exists():
                warp_dst.unlink()
            shutil.move(str(warp_src), str(warp_dst))
        if invwarp_src.exists():
            if invwarp_dst.exists():
                invwarp_dst.unlink()
            shutil.move(str(invwarp_src), str(invwarp_dst))

        # Write 2-channel ImageJ-compatible TIFF overlay (fixed + warped)
        overlay_dst = ants_dir / f"{args.exp_id}_fish{args.fish}_fixed_warped_2ch.tif"
        write_fixed_warped_overlay_tiff(Path(fixed_nii), warped_dst, overlay_dst)

        print(">> Standardized outputs:")
        print("  ", warped_dst)
        print("  ", invwarped_dst)
        print("  ", affine_dst)
        if warp_src.exists():
            print("  ", warp_dst)
        if invwarp_src.exists():
            print("  ", invwarp_dst)
        print("  ", overlay_dst)

        if args.keep_nii:
            keep_dir = ants_dir / "intermediates"
            keep_dir.mkdir(exist_ok=True)
            shutil.copy(fixed_nii,  keep_dir / f"{args.exp_id}_fish{args.fish}_fixed.nii.gz")
            shutil.copy(moving_nii, keep_dir / f"{args.exp_id}_fish{args.fish}_moving.nii.gz")
            print(f">> Kept intermediates in: {keep_dir}")

        print(">> Done. ANTs prefix:", str(prefix))

    finally:
        if not args.keep_nii:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
