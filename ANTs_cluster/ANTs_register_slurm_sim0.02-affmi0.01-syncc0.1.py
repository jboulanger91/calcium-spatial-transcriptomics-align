#!/usr/bin/env python3
"""
ANTs_register_slurm.py

ANTs registration runner for 3D TIFF stacks (timestamped outputs only).

What this script does
---------------------
1) Reads two 3D TIFF stacks ("fixed" and "moving").
2) Converts them to temporary NIfTI files, writing voxel spacing from CLI arguments
   into the NIfTI headers (µm → mm).
3) Runs ANTs `antsRegistration` with:
      --initial-moving-transform [fixed,moving,1]
      Rigid → Similarity
   (Optional blocks for Affine/SyN can be uncommented in `run_ants_registration()`.)
4) Writes only timestamped outputs to avoid collisions when running many sbatch jobs.

Outputs (in --out-dir)
----------------------
All outputs include a timestamp suffix: <exp_id>_fish<fish>_<YYYYmmdd_HHMMSS>_*

- *_warped.nii.gz
- *_inversewarped.nii.gz
- *_overlay.tif
- *_ants_params.json
- ANTs transform files emitted from the timestamped prefix, e.g.:
    *_0GenericAffine.mat

ImageJ overlay TIFF
-------------------
The overlay is written as an ImageJ-compatible hyperstack with axes "TZCYX":
  T=1, Z=slices, C=2 (channel 0=fixed, channel 1=warped moving), Y, X.

Spacing conventions
-------------------
- Voxel spacing is provided explicitly on the command line in microns (µm).
- The script writes these spacings into the temporary NIfTI headers (converted to millimeters, mm) before calling ANTs.
- ANTs/ITK interprets spacing in millimeters (mm). If your spacings are wrong, registration will be wrong.
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile as tiff


def read_tif_write_nii(tif_path: Path, nii_path: Path, spacing_um_xyz: tuple[float, float, float]) -> Path:
    """Read a 3D TIFF stack and write a scalar NIfTI (.nii.gz), forcing spacing into the header."""
    print(f"[I/O] Reading TIFF: {tif_path}")
    img = sitk.ReadImage(str(tif_path), sitk.sitkFloat32)

    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)

    spacing_mm = tuple(float(s) / 1000.0 for s in spacing_um_xyz)
    print(f"[I/O] Using user spacing (um XYZ): {spacing_um_xyz}")
    print(f"[I/O] Writing NIfTI spacing (mm XYZ): {spacing_mm}")
    img.SetSpacing(spacing_mm)

    print(f"[I/O] Writing NIfTI: {nii_path}")
    sitk.WriteImage(img, str(nii_path))

    try:
        img_check = sitk.ReadImage(str(nii_path))
        print(f"[I/O] NIfTI header spacing verified (mm): {img_check.GetSpacing()}")
    except Exception as e:
        print(f"[warn] Could not re-read NIfTI to verify spacing: {e}")

    return nii_path


def _to_uint16_for_imagej(vol_zyx: np.ndarray) -> np.ndarray:
    """Convert a ZYX volume to uint16 for ImageJ overlays (robust display scaling)."""
    if np.issubdtype(vol_zyx.dtype, np.floating):
        p1, p99 = np.percentile(vol_zyx, (1, 99))
        if p99 > p1:
            v = (vol_zyx - p1) / (p99 - p1)
        else:
            v = np.clip(vol_zyx, 0, 1)
        return (np.clip(v, 0, 1) * 65535).astype(np.uint16)
    return np.clip(vol_zyx, 0, 65535).astype(np.uint16)


def run_ants_registration(
    fixed_nii: str,
    moving_nii: str,
    out_prefix: str,
    warped_out: str,
    inverse_warped_out: str,
) -> list[str]:
    """Run ANTs `antsRegistration` and return the exact command list executed."""
    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--interpolation", "Linear",
        "--output", f"[{out_prefix},{warped_out},{inverse_warped_out}]",
        "--write-composite-transform", "1",

        # Center-of-mass initialization
        "--initial-moving-transform", f"[{fixed_nii},{moving_nii},1]",

        # Rigid
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,32,Regular,1]",
        "--convergence", "[1000x500x250x300,1e-6,10]",
        "--smoothing-sigmas", "3x2x1x0",
        "--shrink-factors", "8x4x2x1",

        # Similarity (global isotropic scale ~30%)
        "--transform", "Similarity[0.02]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,32,Regular,1]",
        "--convergence", "[1000x500x250x300,1e-6,10]",
        "--smoothing-sigmas", "3x2x1x0",
        "--shrink-factors", "8x4x2x1",

        # Affine (MI) – conservative, prevents CC-driven weird scaling/shear
        "--transform", "Affine[0.01]",
        "--metric", f"MI[{fixed_nii},{moving_nii},1,64,Regular,1]",
        "--convergence", "[400x200x100x50,1e-6,10]",
        "--smoothing-sigmas", "2x1x0x0",
        "--shrink-factors", "8x4x2x1",

        # SyN (CC) – ONLY tiny, fine-scale polish
        "--transform",        "SyN[0.1,6,0]",
        "--metric",           f"CC[{fixed_nii},{moving_nii},1,2]",
        "--convergence",      "[200x200x200x100,1e-7,10]",
        "--smoothing-sigmas", "4x3x2x1",
        "--shrink-factors",   "12x8x4x2",
    ]

    print(">> Running ANTs:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert 3D TIFF stacks to NIfTI (with user spacing) and run ANTs registration; write timestamped overlay + params JSON."
    )
    p.add_argument("--fixed", required=True, help="Fixed/reference TIFF stack")
    p.add_argument("--moving", required=True, help="Moving TIFF stack")
    p.add_argument("--out-dir", required=True, help="Output folder for results")
    p.add_argument("--exp-id", required=True, help="Experiment id (e.g. exp_001)")
    p.add_argument("--fish", required=True, type=int, help="Fish number (e.g. 2)")
    p.add_argument("--keep-nii", action="store_true", help="Keep temporary NIfTI files under out-dir/intermediates_<timestamp>/")

    p.add_argument("--fixed-spacing-um", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--moving-spacing-um", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))

    args = p.parse_args()

    if shutil.which("antsRegistration") is None:
        sys.exit("Error: antsRegistration not found in PATH. Please install ANTs or update PATH.")

    fixed_tif = Path(args.fixed).expanduser().resolve()
    moving_tif = Path(args.moving).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.exp_id}_fish{args.fish}_{ts}"

    # Timestamped prefix + outputs (no collisions ever)
    prefix = out_dir / f"{run_tag}_"
    warped_out = out_dir / f"{run_tag}_warped.nii.gz"
    inverse_warped_out = out_dir / f"{run_tag}_inversewarped.nii.gz"
    overlay_out = out_dir / f"{run_tag}_overlay.tif"
    params_json_out = out_dir / f"{run_tag}_ants_params.json"

    tmpdir = Path(tempfile.mkdtemp(prefix="tif2nii_"))
    fixed_nii = tmpdir / "fixed.nii.gz"
    moving_nii = tmpdir / "moving.nii.gz"

    try:
        print(">> Converting TIFF -> NIfTI (this may take a moment)...")
        read_tif_write_nii(fixed_tif, fixed_nii, tuple(args.fixed_spacing_um))
        read_tif_write_nii(moving_tif, moving_nii, tuple(args.moving_spacing_um))

        fixed_hdr = sitk.ReadImage(str(fixed_nii))
        moving_hdr = sitk.ReadImage(str(moving_nii))
        print(f"[ANTs] Fixed spacing to ANTs (mm):  {fixed_hdr.GetSpacing()}")
        print(f"[ANTs] Moving spacing to ANTs (mm): {moving_hdr.GetSpacing()}")
        print("[warn] Spacings are from CLI inputs (µm→mm). If they are wrong, registration will be wrong.")

        if args.keep_nii:
            keep_dir = out_dir / f"intermediates_{run_tag}"
            keep_dir.mkdir(exist_ok=True)
            shutil.copy2(str(fixed_nii), keep_dir / f"{run_tag}_fixed.nii.gz")
            shutil.copy2(str(moving_nii), keep_dir / f"{run_tag}_moving.nii.gz")
            print(f"[save] Kept intermediates in: {keep_dir}")

        print(">> Running ANTs registration...")
        ants_cmd = run_ants_registration(
            fixed_nii=str(fixed_nii),
            moving_nii=str(moving_nii),
            out_prefix=str(prefix),
            warped_out=str(warped_out),
            inverse_warped_out=str(inverse_warped_out),
        )

        if not warped_out.exists():
            raise FileNotFoundError(f"Expected warped output not found: {warped_out}")

        # Record JSON with the fully expanded command
        run_command_example = " ".join([
            "python3 ANTs_register_local.py",
            f'--fixed "{fixed_tif}"',
            f'--moving "{moving_tif}"',
            f'--fixed-spacing-um {" ".join(str(x) for x in args.fixed_spacing_um)}',
            f'--moving-spacing-um {" ".join(str(x) for x in args.moving_spacing_um)}',
            f"--exp-id {args.exp_id}",
            f"--fish {args.fish}",
            f'--out-dir "{out_dir}"',
            "--keep-nii" if args.keep_nii else "",
        ]).strip()

        record = {
            "timestamp": ts,
            "exp_id": args.exp_id,
            "fish": args.fish,
            "run_tag": run_tag,
            "fixed_tif": str(fixed_tif),
            "moving_tif": str(moving_tif),
            "fixed_spacing_um_cli": list(args.fixed_spacing_um),
            "moving_spacing_um_cli": list(args.moving_spacing_um),
            "fixed_spacing_mm_written": list(fixed_hdr.GetSpacing()),
            "moving_spacing_mm_written": list(moving_hdr.GetSpacing()),
            "out_dir": str(out_dir),
            "ants_prefix": str(prefix),
            "warped": str(warped_out),
            "inverse_warped": str(inverse_warped_out),
            "overlay_tif": str(overlay_out),
            "ants_cmd": ants_cmd,
            "run_command_example": run_command_example,
        }

        print(f"[save] Writing ANTs params JSON: {params_json_out}")
        with open(params_json_out, "w") as f:
            json.dump(record, f, indent=2)

        # Build ImageJ 2-channel overlay
        print(">> Building ImageJ 2-channel overlay (fixed, warped)...")
        fixed_img = sitk.ReadImage(str(fixed_tif), sitk.sitkFloat32)
        if fixed_img.GetNumberOfComponentsPerPixel() > 1:
            fixed_img = sitk.VectorIndexSelectionCast(fixed_img, 0)
        fixed_arr = sitk.GetArrayFromImage(fixed_img)  # Z,Y,X

        warped_img = sitk.ReadImage(str(warped_out))
        warped_arr = sitk.GetArrayFromImage(warped_img)  # Z,Y,X

        fixed_u16 = _to_uint16_for_imagej(fixed_arr)
        warped_u16 = _to_uint16_for_imagej(warped_arr)

        zmin = min(fixed_u16.shape[0], warped_u16.shape[0])
        stacked = np.stack([fixed_u16[:zmin], warped_u16[:zmin]], axis=1)  # (Z,C,Y,X)
        stacked = stacked[np.newaxis, ...]  # (T,Z,C,Y,X)

        print(f"[save] Writing overlay TIFF: {overlay_out}")
        tiff.imwrite(str(overlay_out), stacked, bigtiff=True, imagej=True, metadata={"axes": "TZCYX"})

        print(">> Done. Outputs:")
        print("   ", warped_out)
        print("   ", inverse_warped_out)
        print("   ", overlay_out)
        print("   ", params_json_out)
        print("   ", f"(ANTs transforms: {prefix}* )")

    finally:
        if not args.keep_nii:
            shutil.rmtree(str(tmpdir), ignore_errors=True)


if __name__ == "__main__":
    main()