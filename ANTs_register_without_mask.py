#!/usr/bin/env python3
"""
ANTs_register_without_mask.py

Clean, minimal, robust ANTs affine registration driver tailored to the user's
pipeline. This script:
  - converts two 3D TIFF stacks to NIfTI (with explicit spacing)
  - runs a three-stage ANTs registration (Rigid -> Similarity -> Affine) using the project's
    preferred parameters (kept verbatim)
  - locates ANTs outputs robustly
  - writes standardized outputs:
      <out_dir>/<exp_id>_fish<fish>_warped.nii.gz
      <out_dir>/<exp_id>_fish<fish>_0GenericAffine.mat   (if found)
      <out_dir>/<exp_id>_fish<fish>_fixed_warped_2ch.tif  (ImageJ hyperstack)

Notes
-----
- ANTs must be on PATH (or set PATH before running this script).
- This script avoids raising when ANTs fails to produce an affine file; it
  warns and continues.
- The ANTs registration parameters (metrics / convergence / smoothing / shrink)
  are preserved from your preferred configuration.

Example usage:
python3 ANTs_register_without_mask.py \
  --fixed "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/fish2/prealigned/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_ch1.tif" \
  --moving "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/2p_stacks/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz.tif" \
  --fixed-spacing-um 1 1 1.0 \
  --moving-spacing-um 1 1 2.0 \
  --exp-id exp_001 \
  --fish 2 \
  --out-dir "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/ANTs_output" \
  --keep-nii  
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

# If you have a local ANTs install you want to prepend, uncomment and edit:
os.environ["PATH"] = "/Users/jonathanboulanger-weill/Packages/install/bin:" + os.environ.get("PATH", "")

def um_to_mm_tuple(um_tuple):
    return tuple(float(x) / 1000.0 for x in um_tuple)


def to_mm_tuple(mm_tuple):
    return tuple(float(x) for x in mm_tuple)


def read_tif_write_nii(tif_path: Path, nii_path: Path, spacing_mm=None):
    """Read TIFF with SimpleITK and write NIfTI setting spacing in mm.

    The function selects the first channel if the TIFF is multi-component.
    """
    print(f"[I/O] Reading TIFF: {tif_path}")
    img = sitk.ReadImage(str(tif_path), sitk.sitkFloat32)

    if img.GetNumberOfComponentsPerPixel() > 1:
        # pick channel 0
        img = sitk.VectorIndexSelectionCast(img, 0)

    if spacing_mm is not None:
        print(f"[I/O] Setting spacing (mm): {spacing_mm}")
        img.SetSpacing(tuple(map(float, spacing_mm)))

    print(f"[I/O] Writing NIfTI: {nii_path}")
    sitk.WriteImage(img, str(nii_path))
    return nii_path


def _to_uint16_for_imagej(vol: np.ndarray) -> np.ndarray:
    if np.issubdtype(vol.dtype, np.floating):
        p1, p99 = np.percentile(vol, (1, 99))
        if p99 > p1:
            vol_rescaled = (vol - p1) / (p99 - p1)
        else:
            vol_rescaled = np.clip(vol, 0, 1)
        vol_uint16 = (np.clip(vol_rescaled, 0, 1) * 65535).astype(np.uint16)
    else:
        vol_uint16 = np.clip(vol, 0, 65535).astype(np.uint16)
    return vol_uint16


def locate_warped_output(prefix: Path, ants_dir: Path) -> Path:
    candidates = [
        prefix.with_name(prefix.name + "Warped.nii.gz"),
        ants_dir / (prefix.name + "Warped.nii.gz"),
        Path.cwd() / (prefix.name + "Warped.nii.gz"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # glob fallback
    for base in (ants_dir, Path.cwd()):
        hits = sorted(base.glob(f"{prefix.name}*Warped.nii.gz"))
        if hits:
            return hits[0]

    raise FileNotFoundError(f"Could not find ANTs warped output with prefix '{prefix.name}' in {ants_dir} or CWD.")


def run_ants_registration(fixed_nii: str, moving_nii: str, out_prefix: str):
    # Keep the ANTs registration parameters (Rigid -> Similarity -> Affine) as requested.
    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--interpolation", "Linear",
        "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz,{out_prefix}InverseWarped.nii.gz]",
        "--write-composite-transform", "1",

        # Rigid
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,64,Regular,1]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--smoothing-sigmas", "3x2x1x0",
        "--shrink-factors", "8x4x2x1",

        # Similarity (adds isotropic scale)
        "--transform", "Similarity[0.1]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,64,Regular,1]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--smoothing-sigmas", "3x2x1x0",
        "--shrink-factors", "8x4x2x1",

        # Affine
        "--transform", "Affine[0.1]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,64,Regular,1]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--smoothing-sigmas", "3x2x1x0",
        "--shrink-factors", "8x4x2x1",

        # Non-linear SyN
        "--transform", "SyN[0.15,3,0]",                 # step=0.15, smoother updates than 6
        "--metric", f"CC[{fixed_nii},{moving_nii},1,4]",# CC radius 4 (more robust, slower)
        "--convergence", "[200x200x150x100x50,1e-7,10]",
        "--smoothing-sigmas", "4x3x2x1x0",
        "--shrink-factors", "10x8x4x2x1",
    ]

    print(
        ">> Running ANTs:",
        " ".join(cmd)
    )
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Convert TIFF -> NIfTI and run ANTs registration (Rigid->Similarity->Affine).")
    p.add_argument("--fixed", required=True, help="Fixed/reference TIFF stack (source TIFF)")
    p.add_argument("--moving", required=True, help="Moving TIFF stack (source TIFF)")
    p.add_argument("--out-dir", required=False, default=None, help="Output folder for results")
    p.add_argument("--exp-id", required=True, help="Experiment id (e.g. exp_001)")
    p.add_argument("--fish", required=True, type=int, help="Fish number (e.g. 2)")

    # spacing in microns (preferred by user workflow) or mm
    p.add_argument("--fixed-spacing-um", nargs=3, type=float, help="Fixed spacing in µm (sx sy sz)")
    p.add_argument("--moving-spacing-um", nargs=3, type=float, help="Moving spacing in µm (sx sy sz)")
    p.add_argument("--fixed-spacing-mm", nargs=3, type=float, help="Fixed spacing in mm (overrides um if given)")
    p.add_argument("--moving-spacing-mm", nargs=3, type=float, help="Moving spacing in mm (overrides um if given)")

    p.add_argument("--keep-nii", action='store_true', help="Keep temporary NIfTI files in output dir/intermediates")

    args = p.parse_args()

    # Check ANTs
    if shutil.which("antsRegistration") is None:
        sys.exit("Error: antsRegistration not found in PATH. Please install ANTs or update PATH.")

    fixed_tif = Path(args.fixed).expanduser().resolve()
    moving_tif = Path(args.moving).expanduser().resolve()

    out_dir = Path(args.out_dir) if args.out_dir else fixed_tif.parent.parent / "ANTs_output"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # standardized names
    warped_final = out_dir / f"{args.exp_id}_fish{args.fish}_warped.nii.gz"
    affine_final = out_dir / f"{args.exp_id}_fish{args.fish}_0GenericAffine.mat"
    overlay_tif = out_dir / f"{args.exp_id}_fish{args.fish}_fixed_warped_2ch.tif"

    # determine spacings to pass to converter (in mm)
    if args.fixed_spacing_mm:
        fixed_spacing_mm = to_mm_tuple(args.fixed_spacing_mm)
    elif args.fixed_spacing_um:
        fixed_spacing_mm = um_to_mm_tuple(args.fixed_spacing_um)
    else:
        fixed_spacing_mm = None

    if args.moving_spacing_mm:
        moving_spacing_mm = to_mm_tuple(args.moving_spacing_mm)
    elif args.moving_spacing_um:
        moving_spacing_mm = um_to_mm_tuple(args.moving_spacing_um)
    else:
        moving_spacing_mm = None

    tmpdir = Path(tempfile.mkdtemp(prefix="tif2nii_"))
    fixed_nii = tmpdir / "fixed.nii.gz"
    moving_nii = tmpdir / "moving.nii.gz"

    try:
        print(">> Converting TIFF -> NIfTI (this may take a moment)...")
        read_tif_write_nii(fixed_tif, fixed_nii, spacing_mm=fixed_spacing_mm)
        read_tif_write_nii(moving_tif, moving_nii, spacing_mm=moving_spacing_mm)

        # ANTs prefix (ANTs will append names like <prefix>Warped.nii.gz)
        prefix = out_dir / f"{args.exp_id}_fish{args.fish}_"

        print(">> Running ANTs registration...")
        run_ants_registration(str(fixed_nii), str(moving_nii), str(prefix))

        print(">> Locating ANTs warped output...")
        warped_src = locate_warped_output(prefix, out_dir)
        print("   found:", warped_src)

        # ANTs already wrote the warped output directly to the canonical filename.
        # Do NOT copy or rename here (macOS case-insensitive FS would delete the source).
        print(f"[save] Warped NIfTI already written by ANTs: {warped_final}")

        # try to find affine transform; don't fail hard if missing
        affine_candidates = list(out_dir.glob(f"{prefix.name}*GenericAffine.mat"))
        affine_candidates += list(out_dir.glob(f"{prefix.name}*Affine.mat"))
        affine_src = affine_candidates[0] if affine_candidates else None
        if affine_src:
            try:
                shutil.copy2(affine_src, affine_final)
                print(f"[save] Copied affine transform: {affine_final}")
            except Exception as e:
                print(f"[warn] Could not copy affine transform: {e}")
        else:
            print(f"[warn] Affine transform not found for prefix '{prefix.name}' — continuing without copying it.")

        # Create ImageJ-ready 2-channel TIFF (fixed, warped)
        print(">> Reading fixed TIFF and warped NIfTI to build ImageJ overlay...")
        # Read fixed TIFF as single-channel volume
        fixed_img = sitk.ReadImage(str(fixed_tif), sitk.sitkFloat32)
        if fixed_img.GetNumberOfComponentsPerPixel() > 1:
            fixed_img = sitk.VectorIndexSelectionCast(fixed_img, 0)
        fixed_arr = sitk.GetArrayFromImage(fixed_img)  # Z,Y,X

        warped_img = sitk.ReadImage(str(warped_final))
        warped_arr = sitk.GetArrayFromImage(warped_img)  # Z,Y,X

        # ensure same dtype/scale for ImageJ display
        fixed_u16 = _to_uint16_for_imagej(fixed_arr)
        warped_u16 = _to_uint16_for_imagej(warped_arr)

        if fixed_u16.shape != warped_u16.shape:
            print(f"[warn] Shape mismatch fixed {fixed_u16.shape} vs warped {warped_u16.shape}. The overlay will be written but may appear misaligned in ImageJ.")

        # Build (T=1, Z, C=2, Y, X)
        # Ensure both volumes have the same Z dimension by trimming/padding the shorter one.
        zmin = min(fixed_u16.shape[0], warped_u16.shape[0])
        fixed_crop = fixed_u16[:zmin]
        warped_crop = warped_u16[:zmin]

        stacked = np.stack([fixed_crop, warped_crop], axis=1)  # (Z, C, Y, X)
        stacked = stacked[np.newaxis, ...]  # (T=1, Z, C, Y, X)

        print(f"[save] Writing ImageJ 2-channel TIFF: {overlay_tif}")
        tiff.imwrite(str(overlay_tif), stacked, bigtiff=True, imagej=True, metadata={"axes": "TZCYX"})

        print(">> All done. Outputs:")
        print("   ", warped_final)
        if affine_src:
            print("   ", affine_final)
        print("   ", overlay_tif)

        if args.keep_nii:
            keep_dir = out_dir / "intermediates"
            keep_dir.mkdir(exist_ok=True)
            shutil.copy2(str(fixed_nii), keep_dir / f"{args.exp_id}_fish{args.fish}_fixed.nii.gz")
            shutil.copy2(str(moving_nii), keep_dir / f"{args.exp_id}_fish{args.fish}_moving.nii.gz")
            print(f"[save] Kept intermediates in: {keep_dir}")

    finally:
        if not args.keep_nii:
            shutil.rmtree(str(tmpdir), ignore_errors=True)


if __name__ == "__main__":
    main()
