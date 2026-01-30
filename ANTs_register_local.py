#!/usr/bin/env python3
"""
ANTs_register_local.py

ANTs registration runner for 3D TIFF stacks.

What this script does
---------------------
1) Reads two 3D TIFF stacks ("fixed" and "moving").
2) Converts them to temporary NIfTI files with user-provided voxel spacing.
3) Runs ANTs `antsRegistration` using a multi-stage pipeline:
     Rigid → Similarity → Affine → SyN
   (parameters are kept verbatim in `run_ants_registration()` so the command is
   easy to audit and compare across runs).
4) Writes standardized outputs into `--out-dir`:
     - <exp_id>_fish<fish>_warped.nii.gz
     - <exp_id>_fish<fish>_inversewarped.nii.gz
     - <exp_id>_fish<fish>_0GenericAffine.mat (copied if ANTs produced one)
     - <exp_id>_fish<fish>_fixed_warped_2ch.tif (latest overlay, overwritten)
     - <exp_id>_fish<fish>_<YYYYmmdd_HHMMSS>_overlay.tif (timestamped overlay)
     - <exp_id>_fish<fish>_<YYYYmmdd_HHMMSS>_ants_params.json (timestamped)
     - <exp_id>_fish<fish>_ants_params_latest.json (latest params, overwritten)

ImageJ overlay TIFF
-------------------
The overlay is written as an ImageJ-compatible hyperstack with axes "TZCYX":
  T=1, Z=slices, C=2 (channel 0=fixed, channel 1=warped moving), Y, X.

Spacing conventions
-------------------
- CLI accepts spacing in microns (µm) via `--*-spacing-um`.
- Internally we convert to millimeters (mm) before writing NIfTI, because
  ANTs/ITK spacing is typically interpreted in mm.

Requirements
------------
- ANTs binaries must be available on PATH (script checks for `antsRegistration`).
- Python packages: SimpleITK, tifffile, numpy.

Example
-------
python3 ANTs_register_without_mask.py \
  --fixed "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/fish2/prealigned/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_ch1.tif" \
  --moving "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/2p_stacks/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz.tif" \
  --fixed-spacing-um 1 1 1.0 \
  --moving-spacing-um 1 1 2.0 \
  --exp-id exp_001 \
  --fish 2 \
  --out-dir "/path/to/ANTs_output" \
  --keep-nii

Notes
-----
- The script is intentionally "thin": it avoids hidden preprocessing so that
  registration changes are driven by explicit ANTs parameters.
- Output naming is standardized to make iteration and comparison easy.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import json
from pathlib import Path
import datetime

import SimpleITK as sitk
import tifffile as tiff
import numpy as np

# If you have a local ANTs install you want to prepend, uncomment and edit:
os.environ["PATH"] = "/Users/jonathanboulanger-weill/Packages/install/bin:" + os.environ.get("PATH", "")

def um_to_mm_tuple(um_tuple):
    """Convert a 3-tuple/list of spacings from microns (µm) to millimeters (mm).

    Parameters
    ----------
    um_tuple : Iterable[float]
        Spacing values in microns, typically (sx, sy, sz).

    Returns
    -------
    tuple[float, float, float]
        Same spacing values expressed in millimeters.
    """
    return tuple(float(x) / 1000.0 for x in um_tuple)


def to_mm_tuple(mm_tuple):
    """Normalize/validate a 3-tuple/list of spacings already expressed in mm."""
    return tuple(float(x) for x in mm_tuple)


def read_tif_write_nii(tif_path: Path, nii_path: Path, spacing_mm=None):
    """Read a TIFF stack and write a NIfTI (.nii.gz) with optional spacing.

    This uses SimpleITK to read the TIFF. If the TIFF is multi-component
    (e.g., RGB or multi-channel), the script selects channel 0 so that ANTs
    receives a single scalar volume.

    Parameters
    ----------
    tif_path : pathlib.Path
        Input TIFF path.
    nii_path : pathlib.Path
        Output NIfTI path.
    spacing_mm : tuple[float, float, float] | None
        Physical voxel spacing (sx, sy, sz) in millimeters. If None, the
        spacing stored in the TIFF/ITK header (if any) is preserved.

    Returns
    -------
    pathlib.Path
        The written NIfTI path.
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
    """Convert a volume to uint16 suitable for ImageJ visualization.

    - If input is float, robustly rescales intensities using the 1st–99th
      percentile to span [0, 65535].
    - If input is integer, clips to [0, 65535].

    This is for display/overlay convenience (not quantitative analysis).
    """
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


def safe_copy2(src: Path, dst: Path) -> None:
    """Copy src → dst, skipping the operation if both refer to the same file.

    This avoids `SameFileError` when ANTs writes an output whose name only
    differs by case (common on case-insensitive filesystems) or when the
    destination already points to the same inode.
    """
    src = Path(src)
    dst = Path(dst)
    try:
        # samefile handles symlinks; may raise if one doesn't exist
        if src.exists() and dst.exists() and src.samefile(dst):
            print(f"[info] Not copying; source and destination are the same file: {dst}")
            return
    except Exception:
        # Fall back to string comparison
        if str(src.resolve()) == str(dst.resolve()):
            print(f"[info] Not copying; source and destination resolve to the same path: {dst}")
            return

    shutil.copy2(src, dst)


def locate_warped_output(prefix: Path, ants_dir: Path) -> Path:
    """Locate the ANTs warped output file produced by `antsRegistration`.

    ANTs may write outputs relative to the provided prefix and/or the current
    working directory. This helper searches a small set of expected locations
    and then falls back to a glob search.

    Parameters
    ----------
    prefix : pathlib.Path
        Output prefix passed to ANTs (e.g., <out_dir>/exp_001_fish2_).
    ants_dir : pathlib.Path
        Directory where outputs are expected (typically `--out-dir`).

    Returns
    -------
    pathlib.Path
        Path to the first matching `*Warped.nii.gz` file.

    Raises
    ------
    FileNotFoundError
        If no warped output can be found.
    """
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


def run_ants_registration(fixed_nii: str, moving_nii: str, out_prefix: str, warped_out: str, inverse_warped_out: str):
    """Run ANTs `antsRegistration` with the project’s multi-stage parameters.

    Parameters
    ----------
    fixed_nii : str
        Fixed/reference NIfTI path.
    moving_nii : str
        Moving NIfTI path.
    out_prefix : str
        ANTs output prefix. ANTs will emit transform files using this prefix.
    warped_out : str
        Explicit path where ANTs should write the warped moving image.
    inverse_warped_out : str
        Explicit path where ANTs should write the inverse-warped image.

    Returns
    -------
    list[str]
        The full command list executed (useful for logging/JSON export).

    Notes
    -----
    The `cmd` list is intentionally explicit so it can be copied/pasted and
    versioned exactly between runs.
    """
    # Keep the ANTs registration parameters (Rigid -> Similarity -> Affine) as requested.
    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--interpolation", "Linear",
        "--output", f"[{out_prefix},{warped_out},{inverse_warped_out}]",
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

        # SyN (conservative)
        #"--transform", "SyN[0.30,2,0]",                # step=0.30 (was 0.25)
        #"--metric",    f"CC[{fixed_nii},{moving_nii},1,8]",
        #"--convergence","[500x400x300x200x150x100,1e-7,10]",
        #"--smoothing-sigmas","6x5x4x3x2x1",
        #"--shrink-factors","32x16x8x6x4x2",
    ]

    print(
        ">> Running ANTs:",
        " ".join(cmd)
    )
    subprocess.run(cmd, check=True)
    return cmd


def main():
    p = argparse.ArgumentParser(
        description=(
            "Convert 3D TIFF stacks to NIfTI (with user spacing) and run ANTs registration "
            "(Rigid → Similarity → Affine → SyN), then write standardized outputs and an ImageJ overlay."
        )
    )
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
    params_json = out_dir / f"{args.exp_id}_fish{args.fish}_ants_params.json"
    inverse_warped_final = out_dir / f"{args.exp_id}_fish{args.fish}_inversewarped.nii.gz"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_tif_ts = out_dir / f"{args.exp_id}_fish{args.fish}_{ts}_overlay.tif"
    params_json_ts = out_dir / f"{args.exp_id}_fish{args.fish}_{ts}_ants_params.json"
    params_json_latest = out_dir / f"{args.exp_id}_fish{args.fish}_ants_params_latest.json"

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
        ants_cmd = run_ants_registration(
            str(fixed_nii),
            str(moving_nii),
            str(prefix),
            str(warped_final),
            str(inverse_warped_final),
        )

        if not warped_final.exists():
            raise FileNotFoundError(f"Expected warped output not found: {warped_final}")

        print(f"[save] Warped NIfTI written: {warped_final}")

        # try to find affine transform; don't fail hard if missing
        affine_candidates = list(out_dir.glob(f"{prefix.name}*GenericAffine.mat"))
        affine_candidates += list(out_dir.glob(f"{prefix.name}*Affine.mat"))
        affine_src = affine_candidates[0] if affine_candidates else None
        if affine_src:
            try:
                safe_copy2(Path(affine_src), Path(affine_final))
                print(f"[save] Copied affine transform: {affine_final}")
            except Exception as e:
                print(f"[warn] Could not copy affine transform: {e}")
        else:
            print(f"[warn] Affine transform not found for prefix '{prefix.name}' — continuing without copying it.")

        # Compose run_command_example string
        run_cmd_parts = [
            "python3 ANTs_register_without_mask.py",
            f'--fixed "{fixed_tif}"',
            f'--moving "{moving_tif}"',
        ]
        if args.fixed_spacing_um:
            run_cmd_parts += ["--fixed-spacing-um"] + [str(x) for x in args.fixed_spacing_um]
        if args.moving_spacing_um:
            run_cmd_parts += ["--moving-spacing-um"] + [str(x) for x in args.moving_spacing_um]
        run_cmd_parts += [
            f"--exp-id {args.exp_id}",
            f"--fish {args.fish}",
            f'--out-dir "{out_dir}"',
        ]
        if args.keep_nii:
            run_cmd_parts.append("--keep-nii")
        run_command_example = " ".join(run_cmd_parts)

        record = {
            "timestamp": ts,
            "exp_id": args.exp_id,
            "fish": args.fish,
            "fixed_tif": str(fixed_tif),
            "moving_tif": str(moving_tif),
            "fixed_spacing_um": list(args.fixed_spacing_um) if args.fixed_spacing_um else None,
            "moving_spacing_um": list(args.moving_spacing_um) if args.moving_spacing_um else None,
            "fixed_spacing_mm": fixed_spacing_mm,
            "moving_spacing_mm": moving_spacing_mm,
            "out_dir": str(out_dir),
            "ants_prefix": str(prefix),
            "warped": str(warped_final),
            "inverse_warped": str(inverse_warped_final),
            "overlay_tif_timestamped": str(overlay_tif_ts),
            "affine_copied_to": str(affine_final) if affine_src else None,
            "ants_cmd": ants_cmd,
            "run_command_example": run_command_example,
        }

        print(f"[save] Writing ANTs params JSON (timestamped): {params_json_ts}")
        with open(params_json_ts, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[save] Writing ANTs params JSON (latest): {params_json_latest}")
        with open(params_json_latest, "w") as f:
            json.dump(record, f, indent=2)

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

        print(f"[save] Writing ImageJ 2-channel TIFF (timestamped): {overlay_tif_ts}")
        tiff.imwrite(str(overlay_tif_ts), stacked, bigtiff=True, imagej=True, metadata={"axes": "TZCYX"})

        print(">> All done. Outputs:")
        print("   ", warped_final)
        if affine_src:
            print("   ", affine_final)
        print("   ", overlay_tif_ts)
        print("   ", params_json_ts)
        print("   ", params_json_latest)

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
