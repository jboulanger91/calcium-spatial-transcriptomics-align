#!/usr/bin/env python
"""
montage_register_prealigned.py

Build a sequential montage from pre-aligned confocal mini-stacks (TIFF) using 2D rigid registration
on an ANNOTATED Z slice (best_z; selected registration channel REG_CHANNEL), then apply that single
2D transform to every Z slice and every channel in the trimmed block before concatenation.

Key ideas
- Input stacks are already pre-aligned for orientation (no flips/rotations here).
- Each TIFF is read into (Z, Y, X, C). Any Time/Frame axis is folded into Z (Z := T*Z).
- For each stack, a trimmed substack is extracted around an annotated slice best_z
  (from section_annotations.tsv). The annotated slice is the registration slice.
- Registration is performed on that annotated Z slice (REG_CHANNEL) against the previous non-damaged block.
- The resulting rigid transform is applied to the full trimmed 3D block (all Z) and all channels.
- Trimmed aligned blocks are concatenated along Z.
- A longest consecutive run of non-damaged section indices is automatically selected for montage.

Channel conventions (0-based)
- ch0: DAPI
- ch1: GCaMP enhanced by GFP immunostaining
- ch2: Vglut enhanced by DsRed immunostaining

Registration uses the annotated best_z slice from ch1 (GCaMP) by default.

Inputs
- A folder of files named: <exp_id>_fish<fish>_s<idx:02d><input_suffix>
- section_annotations.tsv: produced by annotate_damaged_sections.py (preferred)
- damaged_stacks.txt (optional): maintained for backwards compatibility (fallback if TSV missing)

Outputs
- <slug>_<run_tag>_montaged_<metricTag>.tif                : concatenated aligned volume (ImageJ BigTIFF TZCYX)
- <slug>_<run_tag>_montaged_<metricTag>.png                : QC montage of aligned annotated-Z slices per block
- <slug>_<run_tag>_montaged_<metricTag>_GCaMP_ch1.tif       : GCaMP-only volume (single channel, Z pages)

No user interaction.
"""

import os
import re
import sys
import csv
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import ants  # antspyx
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ===================== USER SETTINGS =====================
# Global identifiers used for ALL input/output names
exp_id = "exp_001"   # e.g. "exp_001" or "exp1_110425"
fish   = 1           # integer fish number

# Folder that contains prealigned stacks named like:
#   exp_001_fish1_s01_pre.tif
folder = Path(
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/"
    "exp1_110425/oct_confocal_stacks/fish1/prealigned"
)

# Sections to consider (s01..s25)
indices = list(range(1, 30 + 1))

# Input naming convention
input_suffix = "_pre.tif"

# Annotation table produced by annotate_damaged_sections.py (preferred source of damaged + best_z)
annotations_tsv = folder / "section_annotations.tsv"

# Optional list of stacks to skip (legacy fallback if TSV missing). If entries are basenames,
# this script matches on basename.
damaged_list_file = os.path.join(folder, "damaged_stacks.txt")

# ---- Windowing around annotated best_z ----
# Keep this many Z slices BEFORE and AFTER best_z (inclusive window)
KEEP_Z_BEFORE = 4
KEEP_Z_AFTER  = 4

# Registration choice: "CC" (slow) or "Mattes" (mutual information)
REG_METRIC = "Mattes"                 # "CC" or "Mattes"
# CC params
DS_FACTOR     = 4
CC_ITERS      = (300, 150, 75)
AFF_SAMPLING  = 256
USE_MASKS     = True
# Mattes params
MI_ITERS      = (3000, 1500, 1000)

# Channel index used for registration (0-based indexing)
REG_CHANNEL = 1

# Figure layout
FIG_COLS = 6
FIG_DPI  = 180
# ========================================================


# =============== Small helpers ===============
def to_float01(x):
    x = x.astype(np.float32, copy=False)
    p1, p99 = np.percentile(x, (1, 99)) if np.ptp(x) > 0 else (x.min(), x.max() or 1.0)
    if p99 > p1:
        x = (x - p1) / (p99 - p1)
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return np.clip(x, 0, 1)

def slice_reg(vol_ZYXC, z_index: int, reg_channel: int = None):
    """Return 2D slice at z_index from the requested registration channel."""
    if reg_channel is None:
        reg_channel = REG_CHANNEL
    z_index = int(np.clip(z_index, 0, vol_ZYXC.shape[0] - 1))
    if vol_ZYXC.shape[-1] <= reg_channel:
        raise ValueError(
            f"Requested REG_CHANNEL={reg_channel} but volume has C={vol_ZYXC.shape[-1]} channels"
        )
    return vol_ZYXC[z_index, :, :, reg_channel]

def simple_mask_np(img_np):
    x = img_np.astype(np.float32)
    p99 = np.percentile(x, 99) if np.isfinite(x).all() else np.nanmax(x)
    thr = max(1e-6, 0.05 * p99)
    return (x > thr).astype(np.uint8)

def downsample_ants(img_ants, factor=2):
    ny, nx = img_ants.shape
    new_size = (max(1, ny // factor), max(1, nx // factor))
    return ants.resample_image(img_ants, new_size, use_voxels=True, interp_type=1)

def write_bigtiff_zyxc(path, arr_zyxc, like_dtype):
    """
    Write (Z,Y,X,C) as an ImageJ-compatible BigTIFF hyperstack with axes TZCYX.
    Ensures ImageJ/Fiji interprets Z correctly (T=1, Z>1).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arr = arr_zyxc
    if np.issubdtype(arr.dtype, np.floating) and np.issubdtype(like_dtype, np.integer):
        info = np.iinfo(like_dtype)
        arr = np.clip(arr, info.min, info.max).astype(like_dtype)
    else:
        arr = arr.astype(like_dtype, copy=False)

    Z, Y, X, C = arr.shape
    arr_ij = arr.transpose(0, 3, 1, 2)    # (Z,C,Y,X)
    arr_ij = arr_ij[np.newaxis, ...]      # (T,Z,C,Y,X)

    tiff.imwrite(
        path,
        arr_ij,
        bigtiff=True,
        imagej=True,
        metadata={"axes": "TZCYX"}
    )

def trim_around_best_z(vol_zyxc: np.ndarray, best_z: int, keep_before: int, keep_after: int):
    """
    Trim a (Z,Y,X,C) volume around best_z to [best_z-keep_before, best_z+keep_after] inclusive.
    Returns:
      trimmed_vol (Z',Y,X,C),
      reg_z_trimmed (index within trimmed corresponding to original best_z)
    """
    Z = vol_zyxc.shape[0]
    best_z = int(np.clip(best_z, 0, Z - 1))

    z0 = max(0, best_z - int(keep_before))
    z1 = min(Z - 1, best_z + int(keep_after))  # inclusive

    trimmed = vol_zyxc[z0:z1 + 1]
    reg_z_trimmed = best_z - z0
    return trimmed, reg_z_trimmed

def read_section_annotations(tsv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict keyed by both absolute path string and basename (if possible).
    Values: {"damaged": bool, "best_z": Optional[int]}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not tsv_path.exists():
        return out

    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            p = (row.get("path") or "").strip()
            if not p:
                continue
            bn = os.path.basename(p)
            damaged = int((row.get("damaged") or "0").strip() or 0) == 1
            best_z_str = (row.get("best_z") or "").strip()
            best_z = int(best_z_str) if best_z_str != "" else None

            out[p] = {"damaged": damaged, "best_z": best_z}
            out[bn] = {"damaged": damaged, "best_z": best_z}

    return out
# ============================================


# ============================= Robust reader (T -> Z) =============================

def _from_axes(arr: np.ndarray, axes: str) -> np.ndarray:
    """
    Normalize any axes string (OME or ImageJ-style) to (Z,Y,X,C) with Z := T*Z (frames*slices).
    """
    A = arr
    Ax = [c.upper() for c in axes] if axes else []

    posT = [i for i,a in enumerate(Ax) if a == 'T']
    posZ = [i for i,a in enumerate(Ax) if a == 'Z']
    posY = [i for i,a in enumerate(Ax) if a == 'Y']
    posX = [i for i,a in enumerate(Ax) if a == 'X']
    posC = [i for i,a in enumerate(Ax) if a in ('C','S')]
    posOther = [i for i,a in enumerate(Ax) if a not in ('T','Z','Y','X','C','S')]

    if not posY or not posX:
        nd = A.ndim
        if nd < 2:
            raise RuntimeError(f"Cannot infer Y/X for axes='{axes}' shape={A.shape}")
        lead = list(range(0, nd-2))
        A = np.transpose(A, lead + [nd-2, nd-1])
        Ax = ['?']*(nd-2) + ['Y','X']
        posT = posZ = posC = posOther = []

    perm = posT + posZ + posOther + posY + posX + posC
    used = set(perm)
    perm += [i for i in range(A.ndim) if i not in used]
    A = np.transpose(A, perm)

    ntzo = len(posT) + len(posZ) + len(posOther)
    y_dim = A.shape[ntzo]
    x_dim = A.shape[ntzo+1]
    tzo_block = A.shape[:ntzo] if ntzo > 0 else (1,)
    c_block   = A.shape[ntzo+2: ntzo+2+len(posC)] if len(posC) > 0 else (1,)
    leftovers = A.shape[ntzo+2+len(posC):]

    Z = int(np.prod(tzo_block)) if np.prod(tzo_block) else 1
    C = int(np.prod(c_block))   if np.prod(c_block)   else 1
    L = int(np.prod(leftovers)) if leftovers else 1

    Z_total = Z * L
    out = A.reshape((Z_total, y_dim, x_dim, C))
    return out

def _guess_layout(arr: np.ndarray) -> np.ndarray:
    A = arr
    nd = A.ndim

    if nd == 3 and A.shape[-1] <= 8:
        return A[None, ...]  # (1,Y,X,C)

    if nd == 4:
        if A.shape[-1] <= 8:
            return A
        if A.shape[1] <= 8:
            return np.transpose(A, (0, 2, 3, 1))
        if A.shape[0] <= 8:
            return np.transpose(A, (3, 1, 2, 0))

    raise RuntimeError("Could not guess layout from array alone.")

def read_stack_ZYXC(path: str):
    """
    Read TIFF into (Z,Y,X,C) with Z := T*Z.
    Returns: (zyxc_float32, like_dtype)
    """
    with tiff.TiffFile(path) as tf:
        s = tf.series[0]
        arr = s.asarray()
        axes = getattr(s, "axes", "") or ""
        like_dtype = arr.dtype

        if axes:
            try:
                zyxc = _from_axes(arr, axes)
                return zyxc.astype(np.float32, copy=False), like_dtype
            except Exception as e:
                print(f"[warn] axes='{axes}' failed on {os.path.basename(path)}: {e}")

        try:
            zyxc = _guess_layout(arr)
            return zyxc.astype(np.float32, copy=False), like_dtype
        except Exception:
            n_pages = len(tf.pages)
            first = tf.pages[0].asarray()
            if first.ndim == 3 and first.shape[-1] <= 8:
                Y, X, C = first.shape
                out = np.empty((n_pages, Y, X, C), dtype=np.float32)
                out[0] = first
                for z in range(1, n_pages):
                    out[z] = tf.pages[z].asarray()
                return out, like_dtype
            else:
                Y, X = first.shape[:2]
                out = np.empty((n_pages, Y, X, 1), dtype=np.float32)
                out[0, :, :, 0] = first if first.ndim == 2 else first[..., 0]
                for z in range(1, n_pages):
                    p = tf.pages[z].asarray()
                    out[z, :, :, 0] = p if p.ndim == 2 else p[..., 0]
                return out, like_dtype
# =====================================================================


# =============== Registration & application ===============
def register_2d(fixed_ref_np, moving_np, metric="CC",
                ds_factor=4, cc_iters=(300,150,75), aff_sampling=256,
                use_masks=True, mi_iters=(1000,500,250)):
    """Estimate rigid 2D transform; return (fwdtransforms, fixed_fullres_ants)."""
    fixed_ref = ants.from_numpy(to_float01(fixed_ref_np))
    moving    = ants.from_numpy(to_float01(moving_np))

    if metric == "CC":
        mask_kwargs = {}
        if use_masks:
            fixed_mask  = ants.from_numpy(simple_mask_np(fixed_ref.numpy()))
            moving_mask = ants.from_numpy(simple_mask_np(moving.numpy()))
            mask_kwargs = {"mask": fixed_mask, "moving_mask": moving_mask}

        fixed_ds  = downsample_ants(fixed_ref, factor=ds_factor)
        moving_ds = downsample_ants(moving,     factor=ds_factor)
        if use_masks:
            mask_kwargs = {
                "mask":        downsample_ants(mask_kwargs["mask"],        factor=ds_factor),
                "moving_mask": downsample_ants(mask_kwargs["moving_mask"], factor=ds_factor),
            }
        try:
            reg = ants.registration(
                fixed=fixed_ds, moving=moving_ds, type_of_transform="Rigid",
                aff_metric="CC", aff_sampling=aff_sampling,
                reg_iterations=cc_iters, verbose=False, **mask_kwargs
            )
        except Exception:
            reg = ants.registration(
                fixed=fixed_ds, moving=moving_ds, type_of_transform="Rigid",
                aff_metric="CC", reg_iterations=cc_iters, verbose=False, **mask_kwargs
            )
        return reg["fwdtransforms"], fixed_ref

    reg = ants.registration(
        fixed=fixed_ref, moving=moving, type_of_transform="Rigid",
        aff_metric="mattes", reg_iterations=mi_iters, verbose=False
    )
    return reg["fwdtransforms"], fixed_ref

def ants_apply_rigid_2d_to_stack(vol_ZYXC, fwdtransforms, fixed2d_ref):
    """Apply the same 2D rigid to ALL (Z,C) slices; returns float32 (Z,Y,X,C)."""
    Z, Y, X, C = vol_ZYXC.shape
    out = np.empty_like(vol_ZYXC, dtype=np.float32)
    for z in range(Z):
        for c in range(C):
            mov = ants.from_numpy(vol_ZYXC[z, :, :, c])
            moved = ants.apply_transforms(
                fixed=fixed2d_ref, moving=mov, transformlist=fwdtransforms, interpolator="linear"
            )
            out[z, :, :, c] = moved.numpy().astype(np.float32)
    return out
# =====================================================================


# =============== Centers figure ===============
def save_centers_figure(centers_triplets, out_png_path, cols=6, dpi=180):
    """centers_triplets: [(idx, img01, basename), ...], sorted by idx."""
    centers_triplets = sorted(centers_triplets, key=lambda x: x[0])

    n = len(centers_triplets)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    fig_w = 3.2 * cols
    fig_h = 3.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        idx, img, name = centers_triplets[i]
        ax = axes[i]
        ax.imshow(img, cmap='gray', interpolation='nearest')
        ax.set_title(f"{idx}: {name}", fontsize=8)
        ax.axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
# =====================================================================


# =========================== DRIVER ==================================
if __name__ == "__main__":
    ann = read_section_annotations(annotations_tsv)
    if ann:
        print(f"[ann] loaded annotations from: {annotations_tsv}")
    else:
        print(f"[ann] no annotations found at: {annotations_tsv} (will fall back to damaged_stacks.txt and best_z=middle)")

    # Collect actual files present for the requested indices
    present: Dict[int, str] = {}
    for i in indices:
        p = os.path.join(folder, f"{exp_id}_fish{fish}_s{i:02d}{input_suffix}")
        if os.path.exists(p):
            present[i] = p
        else:
            print(f"⚠️ missing: {p}")
    if not present:
        sys.exit("No stacks found.")

    # Legacy damaged list (fallback)
    damaged_names = set()
    if os.path.exists(damaged_list_file):
        with open(damaged_list_file, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    damaged_names.add(os.path.basename(s))
        print(f"[damaged] loaded {len(damaged_names)} entries from damaged_stacks.txt")
    else:
        print("[damaged] no damaged_stacks.txt found")

    def is_damaged(path_str: str) -> bool:
        bn = os.path.basename(path_str)
        if bn in ann:
            return bool(ann[bn]["damaged"])
        return bn in damaged_names

    def get_best_z(path_str: str, Z: int) -> int:
        bn = os.path.basename(path_str)
        bz = ann.get(bn, {}).get("best_z", None)
        if bz is None:
            return int(Z // 2)
        return int(np.clip(int(bz), 0, Z - 1))

    # Build list of good indices (present and not damaged)
    good_indices = [i for i in sorted(present.keys()) if not is_damaged(present[i])]

    def longest_consecutive_run(sorted_ints):
        best_start = best_end = None
        cur_start = cur_end = None
        for v in sorted_ints:
            if cur_start is None:
                cur_start = cur_end = v
            elif v == cur_end + 1:
                cur_end = v
            else:
                if best_start is None or (cur_end - cur_start) > (best_end - best_start):
                    best_start, best_end = cur_start, cur_end
                cur_start = cur_end = v
        if cur_start is not None:
            if best_start is None or (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
        return (best_start, best_end) if best_start is not None else (None, None)

    run_start, run_end = longest_consecutive_run(good_indices)
    if run_start is None:
        sys.exit("No non-damaged consecutive run found to montage.")

    all_paths: List[Tuple[int, str]] = []
    for i in range(run_start, run_end + 1):
        p = present.get(i)
        if p is None:
            print(f"[warn] expected present file for index {i} but missing")
            continue
        all_paths.append((i, p))

    print(f"[info] Selected longest non-damaged consecutive run: {run_start}..{run_end} -> {len(all_paths)} stacks")
    print(f"[info] Windowing around best_z: keep_before={KEEP_Z_BEFORE}, keep_after={KEEP_Z_AFTER} (inclusive)")

    blocks = []
    centers_triplets = []
    base_dtype = None

    prev_center = None
    prev_ref    = None

    metric_tag = "CCfast" if REG_METRIC == "CC" else "MattesMI"

    for i, p in all_paths:
        name = os.path.basename(p)
        if is_damaged(p):
            print(f"[skip] damaged: {name}")
            continue

        print(f"\n[align] {name} (index {i})")
        vol, like_dtype = read_stack_ZYXC(p)
        if base_dtype is None:
            base_dtype = like_dtype

        best_z_full = get_best_z(p, vol.shape[0])
        if (os.path.basename(p) not in ann) or (ann.get(os.path.basename(p), {}).get("best_z", None) is None):
            print(f"[warn] no best_z in annotations for {name}; using middle z={best_z_full}")
        else:
            print(f"[info] best_z (annotated) for {name}: {best_z_full}")

        vol_trim, reg_z = trim_around_best_z(vol, best_z_full, KEEP_Z_BEFORE, KEEP_Z_AFTER)
        if vol_trim.shape[0] < 1:
            print(f"[warn] skipping {name}: empty after trimming")
            continue

        moving_reg = slice_reg(vol_trim, reg_z).astype(np.float32)

        if prev_ref is None:
            blocks.append(vol_trim.astype(np.float32))
            centers_triplets.append((i, to_float01(moving_reg), name))
            prev_center = moving_reg
            prev_ref    = ants.from_numpy(prev_center)
            print("[base] set as reference for subsequent stacks")
            continue

        if REG_METRIC == "CC":
            fwd, _ = register_2d(
                fixed_ref_np=prev_center, moving_np=moving_reg, metric="CC",
                ds_factor=DS_FACTOR, cc_iters=CC_ITERS, aff_sampling=AFF_SAMPLING, use_masks=USE_MASKS
            )
        else:
            fwd, _ = register_2d(
                fixed_ref_np=prev_center, moving_np=moving_reg, metric="Mattes", mi_iters=MI_ITERS
            )

        aligned = ants_apply_rigid_2d_to_stack(vol_trim, fwd, fixed2d_ref=prev_ref)
        blocks.append(aligned)
        print(f"[concat] appended trimmed+aligned block shape: {aligned.shape} (reg_z={reg_z})")

        aligned_reg = slice_reg(aligned, reg_z).astype(np.float32)
        centers_triplets.append((i, to_float01(aligned_reg), name))
        prev_center = aligned_reg
        prev_ref    = ants.from_numpy(prev_center)

    if not blocks:
        sys.exit("Nothing to concatenate (all stacks missing or damaged or trimmed out).")

    master = np.concatenate(blocks, axis=0)
    print("\n[concat] final (Z,Y,X,C) shape:", master.shape)

    slug = f"{exp_id}_fish{fish}"
    tag  = metric_tag
    run_tag = f"s{run_start:02d}-s{run_end:02d}"

    final_out = os.path.join(folder, f"{slug}_{run_tag}_montaged_{tag}.tif")
    write_bigtiff_zyxc(final_out, master, like_dtype=base_dtype)

    if master.shape[-1] >= 2:
        gcamp_out = os.path.join(folder, f"{slug}_{run_tag}_montaged_{tag}_GCaMP_ch1.tif")
        master_gcamp = master[:, :, :, 1:2]
        write_bigtiff_zyxc(gcamp_out, master_gcamp, like_dtype=base_dtype)
    else:
        gcamp_out = None
        print("[warn] master has <2 channels; skipping GCaMP export")

    centers_png = os.path.join(folder, f"{slug}_{run_tag}_montaged_{tag}.png")
    save_centers_figure(centers_triplets, centers_png, cols=FIG_COLS, dpi=FIG_DPI)

    print("[save]")
    print(" ", final_out)
    if gcamp_out is not None:
        print(" ", gcamp_out)
    print(" ", centers_png)
    print("✅ Done.")