#!/usr/bin/env python3
"""
tiff_to_nii_inspect.py

Minimal TIFF → NIfTI converter that:
- reads a TIFF stack
- writes a .nii.gz with the SAME basename
- prints voxel spacing, voxel size, and physical extent

Example usage:
python3 tiff_to_nii_inspect.py \
  --tif "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/2p_stacks/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz.tif" \
  --out-dir "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/2p_stacks" \
  --spacing-um 0.396 0.396 2.0

python3 tiff_to_nii_inspect.py \
--tif "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/fish2/prealigned/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_ch1.tif" \
--out-dir "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/fish2/prealigned" \
--spacing-um 0.621 0.621 1.0
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile as tiff
import SimpleITK as sitk


def read_tiff_as_zyx(path: Path) -> np.ndarray:
    """Read TIFF and return scalar volume as (Z, Y, X)."""
    with tiff.TiffFile(str(path)) as tf:
        s = tf.series[0]
        axes = (getattr(s, "axes", "") or "").upper()
        arr = s.asarray()

    if "T" in axes:
        arr = np.take(arr, 0, axis=axes.index("T"))
        axes = axes.replace("T", "")

    if "C" in axes:
        arr = np.take(arr, 0, axis=axes.index("C"))
        axes = axes.replace("C", "")

    if all(ax in axes for ax in ("Z", "Y", "X")):
        z, y, x = axes.index("Z"), axes.index("Y"), axes.index("X")
        arr = np.moveaxis(arr, [z, y, x], [0, 1, 2])
    else:
        if arr.ndim != 3:
            raise ValueError(f"Cannot interpret axes '{axes}' with shape {arr.shape}")

    return arr.astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True, help="Input TIFF stack")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--spacing-um",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        required=True,
        help="Voxel spacing in microns (X Y Z)",
    )
    args = ap.parse_args()

    tif_path = Path(args.tif).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAME BASENAME, DIFFERENT EXTENSION
    out_nii = out_dir / (tif_path.stem + ".nii.gz")

    vol_zyx = read_tiff_as_zyx(tif_path)
    img = sitk.GetImageFromArray(vol_zyx)

    sx, sy, sz = args.spacing_um
    img.SetSpacing((sx / 1000.0, sy / 1000.0, sz / 1000.0))

    size = np.array(img.GetSize())
    spacing = np.array(img.GetSpacing())
    extent = size * spacing

    print("=== TIFF → NIfTI INSPECTION ===")
    print("Input TIFF:", tif_path)
    print("Output NIfTI:", out_nii)
    print("Numpy shape (Z,Y,X):", vol_zyx.shape)
    print("Voxel spacing (mm):", spacing)
    print("Volume size (vox):", size)
    print("Physical extent (mm):", extent)
    print("================================")

    sitk.WriteImage(img, str(out_nii))
    print("[save]", out_nii)


if __name__ == "__main__":
    main()