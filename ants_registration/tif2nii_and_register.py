#!/usr/bin/env python3
import argparse, os, subprocess, sys, shutil, tempfile
import SimpleITK as sitk
import os
os.environ["PATH"] = "/Users/jonathanboulanger-weill/Packages/install/bin:" + os.environ["PATH"]
import ants  # antspyx

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
        "--float", "0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--interpolation", "Linear",
        "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz,{out_prefix}InverseWarped.nii.gz]",
        # "--use-estimate-learning-rate-once", "1",   # <-- REMOVE THIS LINE
        "--write-composite-transform", "1",

        # Rigid
        "--transform", "Rigid[0.2]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,64,Random,0.3]",
        "--convergence", "[1200x600x300x0,1e-8,10]",
        "--smoothing-sigmas", "6x3x1x0mm",
        "--shrink-factors", "8x4x2x1",

        # Affine
        "--transform", "Affine[0.1]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,32,Regular,0.25]",
        "--convergence", "[200x200x200x100,1e-8,10]",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--shrink-factors", "8x4x2x1",

        # SyN
        #"--transform", "SyN[0.1,6,0]",
        #"--metric", f"CC[{fixed_nii},{moving_nii},1,2]",
        #"--convergence", "[200x200x200x100,1e-7,10]",
        #"--smoothing-sigmas", "4x3x2x1vox",
        #"--shrink-factors", "12x8x4x2",
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
        run_ants_registration(fixed_nii, moving_nii, args.out_prefix)

        if args.keep_nii:
            keep_dir = os.path.abspath(f"{args.out_prefix}intermediates")
            os.makedirs(keep_dir, exist_ok=True)
            shutil.copy(fixed_nii,  os.path.join(keep_dir, "fixed.nii.gz"))
            shutil.copy(moving_nii, os.path.join(keep_dir, "moving.nii.gz"))
            print(f">> Kept intermediates in: {keep_dir}")

        print(">> Done. Key outputs with prefix:", args.out_prefix)
        print(f"  {args.out_prefix}Warped.nii.gz        # moving → fixed")
        print(f"  {args.out_prefix}InverseWarped.nii.gz  # fixed → moving")
        print(f"  {args.out_prefix}0GenericAffine.mat")
        print(f"  {args.out_prefix}1Warp.nii.gz / {args.out_prefix}1InverseWarp.nii.gz")

    finally:
        if not args.keep_nii:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()


#python3 tif2nii_and_register.py \
#  --fixed "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/spatial_transcriptomics/exp1_110425/2p_stacks/2025-10-14_11-59-26_fish004_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_z1-40.tif" \
#  --moving "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/spatial_transcriptomics/exp1_110425/oct_confocal_stacks/fish4_tifs/prealigned_rc/output_filtered_flipped_z23-z137.tif" \
#  --fixed-spacing-um 0.396 0.396 2.0 \
#  --moving-spacing-um 0.32 0.32 1.0 \
#  --out-prefix reg_ \
#  --keep-nii