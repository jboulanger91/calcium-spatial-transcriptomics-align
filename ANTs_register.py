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
  --fixed-spacing-um 1 1 1.0 \
  --moving-spacing-um 0.9 0.9 2.0\
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
import argparse, os, subprocess, sys, shutil, tempfile
import SimpleITK as sitk
import os
os.environ["PATH"] = "/Users/jonathanboulanger-weill/Packages/install/bin:" + os.environ["PATH"]
import ants  # antspyx
from pathlib import Path

def short_name_from_fish(path: Path) -> str:
    """
    Return filename stem starting from 'fish' if present,
    otherwise return the full stem.
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

# --- in run_ants_registration: remove the unsupported flag ---
def run_ants_registration(fixed_nii, moving_nii, out_prefix):
    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        #"--winsorize-image-intensities", "[0.005,0.995]",
        "--interpolation", "Linear",
        "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz,{out_prefix}InverseWarped.nii.gz]",
        "--write-composite-transform", "1",

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

        # SyN
        "--transform", "SyN[0.1,6,0]",
        "--metric", f"CC[{fixed_nii},{moving_nii},1,2]",
        "--convergence", "[200x200x200x100,1e-7,10]",
        "--smoothing-sigmas", "4x3x2x1",
        "--shrink-factors", "12x8x4x2",
    ]
    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser(description="Convert two TIFF stacks to NIfTI with independent spacings and register with ANTs.")
    p.add_argument("--fixed", required=True, help="Fixed/reference TIFF stack")
    p.add_argument("--moving", required=True, help="Moving TIFF stack")
    p.add_argument("--out-prefix", default="reg_", help="Output prefix (default: reg_)")

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

    p.add_argument("--keep-nii", action="store_true", help="Keep intermediate NIfTI files")
    args = p.parse_args()

    if shutil.which("antsRegistration") is None:
        sys.exit("Error: antsRegistration not found in PATH.")

    # Create fixed ANTs output directory
    ants_dir = Path(
        "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
        "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/"
        "data/exp1_110425/ANTs_output"
    ).resolve()
    ants_dir.mkdir(parents=True, exist_ok=True)

    fixed_path = Path(args.fixed).resolve()
    moving_path = Path(args.moving).resolve()

    # Build explicit output prefix: <moving>_to_<fixed>_ (shortened from 'fish')
    moving_short = short_name_from_fish(moving_path)
    fixed_short  = short_name_from_fish(fixed_path)
    out_prefix = ants_dir / f"{moving_short}_to_{fixed_short}_"

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
        run_ants_registration(fixed_nii, moving_nii, str(out_prefix))

        if args.keep_nii:
            keep_dir = ants_dir / "intermediates"
            os.makedirs(keep_dir, exist_ok=True)
            shutil.copy(fixed_nii,  keep_dir / "fixed.nii.gz")
            shutil.copy(moving_nii, keep_dir / "moving.nii.gz")
            print(f">> Kept intermediates in: {keep_dir}")

        print(">> Done. Key outputs with prefix:", out_prefix)
        print(f"  {out_prefix}Warped.nii.gz        # moving → fixed")
        print(f"  {out_prefix}InverseWarped.nii.gz  # fixed → moving")
        print(f"  {out_prefix}0GenericAffine.mat")
        print(f"  {out_prefix}1Warp.nii.gz / {out_prefix}1InverseWarp.nii.gz")

    finally:
        if not args.keep_nii:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
