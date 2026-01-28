#!/usr/bin/env python
"""
montage_register_prealigned.py

Build a sequential montage from pre-aligned confocal mini-stacks (TIFF) using 2D rigid registration
on the middle Z slice (the selected registration channel (REG_CHANNEL)), then apply that single 2D transform to every Z slice and every
channel in the current block before concatenation.

Key ideas
- Input stacks are already pre-aligned for orientation (no flips/rotations here).
- Each TIFF is read into (Z, Y, X, C). Any Time/Frame axis is folded into Z (Z := T*Z).
- Registration is performed on the middle Z slice (REG_CHANNEL) against the previous *non-damaged* block.
- The resulting rigid transform is applied to the full 3D block (all Z) and all channels.
- Blocks are concatenated along Z, with optional per-block trimming (SKIP_Z_TOP/BOTTOM) to hide seams.
- A longest consecutive run of non-damaged section indices is automatically selected for montage.

Channel conventions (0-based)
- ch0: DAPI
- ch1: GCaMP enhanced by GFP immunostaining
- ch2: Vglut enhanced by DsRed immunostaining

Registration uses the middle Z slice from ch1 (GCaMP), which provides the strongest and most reliable signal.

Inputs
- A folder of files named: <prefix><index><input_suffix>, e.g.  ...fish2-s1-10_preRC.tif
- damaged_stacks.txt (optional): list of stacks to skip (one path per line)

Outputs
- <prefix>_FINAL_concat_<metricTag>.tif  : concatenated aligned volume (Z pages, channels interleaved)
- <prefix>_aligned_centers_<metricTag>.png : QC montage of aligned middle-Z slices per block
- <prefix>_FINAL_concat_<metricTag>_GCaMP_ch1.tif : GCaMP-only volume (single channel, Z pages)

No user interaction.
"""

import os
import re
import sys
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import ants  # antspyx
from pathlib import Path
from typing import Optional, Tuple, List

# ===================== USER SETTINGS =====================
# Global identifiers used for ALL input/output names
exp_id = "exp_001"   # e.g. "exp_001" or "exp1_110425"
fish   = 2           # integer fish number

# Folder that contains prealigned stacks named like:
#   exp_001_fish2_s01_pre.tif
folder = Path(
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/"
    "exp1_110425/oct_confocal_stacks/fish2/prealigned"
)

# Sections to consider (s01..s25)
indices = list(range(1, 25 + 1))

# Input naming convention
# Example: exp_001_fish2_s01_pre.tif
input_suffix = "_pre.tif"

# Optional list of stacks to skip (one path per line). If entries are basenames,
# this script matches on basename.
damaged_list_file = os.path.join(folder, "damaged_stacks.txt")

# How many Z slices to drop from each ministack before concatenation.
# Useful to remove duplicated/low-quality edge slices at boundaries.
# These are applied to every ministack; set to 0 to disable.
SKIP_Z_TOP    = 6   # drop this many Z slices from the *start* of each ministack
SKIP_Z_BOTTOM = 6   # drop this many Z slices from the *end* of each ministack

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
# ch0: DAPI
# ch1: GCaMP (GFP immunostaining)
# ch2: Vglut (DsRed immunostaining)
REG_CHANNEL = 1

# Figure layout
FIG_COLS = 6
FIG_DPI  = 180
# ========================================================


# =============== Small helpers ===============
def sanitize_prefix(prefix: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', prefix).strip('_') or "stack"

def to_float01(x):
    x = x.astype(np.float32, copy=False)
    p1, p99 = np.percentile(x, (1,99)) if np.ptp(x) > 0 else (x.min(), x.max() or 1.0)
    if p99 > p1: x = (x - p1) / (p99 - p1)
    else:        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return np.clip(x, 0, 1)

def center_slice_reg(vol_ZYXC, reg_channel: int = None):
    """Return middle-Z 2D slice from the requested registration channel."""
    if reg_channel is None:
        reg_channel = REG_CHANNEL
    zmid = vol_ZYXC.shape[0] // 2
    if vol_ZYXC.shape[-1] <= reg_channel:
        raise ValueError(
            f"Requested REG_CHANNEL={reg_channel} but volume has C={vol_ZYXC.shape[-1]} channels"
        )
    return vol_ZYXC[zmid, :, :, reg_channel]

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

    # Cast/clip to like_dtype without rescaling
    arr = arr_zyxc
    if np.issubdtype(arr.dtype, np.floating) and np.issubdtype(like_dtype, np.integer):
        info = np.iinfo(like_dtype)
        arr = np.clip(arr, info.min, info.max).astype(like_dtype)
    else:
        arr = arr.astype(like_dtype, copy=False)

    # Convert (Z,Y,X,C) -> (T=1, Z, C, Y, X)
    Z, Y, X, C = arr.shape
    arr_ij = arr.transpose(0, 3, 1, 2)   # (Z,C,Y,X)
    arr_ij = arr_ij[np.newaxis, ...]      # (T,Z,C,Y,X)

    tiff.imwrite(
        path,
        arr_ij,
        bigtiff=True,
        imagej=True,
        metadata={"axes": "TZCYX"}
    )
# ============================================


# ============================= Robust reader (T -> Z) =============================

def _from_axes(arr: np.ndarray, axes: str) -> np.ndarray:
    """
    Normalize any axes string (OME or ImageJ-style) to (Z,Y,X,C) with Z := T*Z (frames*slices).
    - Merges all T-like and Z-like dims into one leading Z
    - Keeps Y,X
    - Merges any C/S (channel/sample) dims into trailing C
    """
    A = arr
    Ax = [c.upper() for c in axes] if axes else []

    # Map positions
    posT = [i for i,a in enumerate(Ax) if a == 'T']
    posZ = [i for i,a in enumerate(Ax) if a == 'Z']
    posY = [i for i,a in enumerate(Ax) if a == 'Y']
    posX = [i for i,a in enumerate(Ax) if a == 'X']
    posC = [i for i,a in enumerate(Ax) if a in ('C','S')]  # channel/sample
    posOther = [i for i,a in enumerate(Ax) if a not in ('T','Z','Y','X','C','S')]

    if not posY or not posX:
        # If Y/X not explicitly present, assume last two dims are Y,X
        # (common when axes string is missing or partial)
        # We will reorder by placing presumed Y,X to the end.
        nd = A.ndim
        if nd < 2:
            raise RuntimeError(f"Cannot infer Y/X for axes='{axes}' shape={A.shape}")
        # Build a fake order: [all-but-last-two, Y, X]
        lead = list(range(0, nd-2))
        A = np.transpose(A, lead + [nd-2, nd-1])
        # Now define Ax to match this choice
        Ax = ['?']*(nd-2) + ['Y','X']
        posT = posZ = posC = posOther = []

    # Build desired permutation to group dims: [T... Z... other...] Y X [C...]
    perm = posT + posZ + posOther + posY + posX + posC
    # Add any dims not mentioned (robustness)
    used = set(perm)
    perm += [i for i in range(A.ndim) if i not in used]
    A = np.transpose(A, perm)
    # Rebuild pseudo-axes in that exact order
    Ax_perm = (['T']*len(posT) + ['Z']*len(posZ) + ['?']*len(posOther) +
               ['Y']*len(posY) + ['X']*len(posX) + ['C']*len(posC) +
               ['?']*(A.ndim - len(perm)))

    # Count grouped blocks
    ntzo = len(posT) + len(posZ) + len(posOther)           # will become Z (product)
    hasY = 1
    hasX = 1
    nc   = len(posC)                                       # will become C (product)

    if A.ndim < ntzo + hasY + hasX + nc:
        # Highly unusual, fallback later
        raise RuntimeError(f"Axes mismatch for '{axes}' shape={arr.shape}")

    # Extract dims
    y_dim = A.shape[ntzo]
    x_dim = A.shape[ntzo+1]
    tzo_block = A.shape[:ntzo] if ntzo > 0 else (1,)
    c_block   = A.shape[ntzo+2: ntzo+2+nc] if nc > 0 else (1,)
    leftovers = A.shape[ntzo+2+nc:]

    Z = int(np.prod(tzo_block)) if np.prod(tzo_block) else 1
    C = int(np.prod(c_block))   if np.prod(c_block)   else 1
    L = int(np.prod(leftovers)) if leftovers else 1

    Z_total = Z * L
    out = A.reshape((Z_total, y_dim, x_dim, C))
    return out

def _guess_layout(arr: np.ndarray) -> np.ndarray:
    """
    Heuristics when no axes are available.
    Try common confocal exports and your prealignment saver:
      - (T,Y,X,C) or (Z,Y,X,C)   (last dim small -> C)
      - (T,C,Y,X) or (Z,C,Y,X)
      - (Y,X,C) (single page)
      - pages-as-Z grayscale
    """
    A = arr
    nd = A.ndim

    # (Y,X,C) single frame
    if nd == 3 and A.shape[-1] <= 8:
        return A[None, ...]  # (1,Y,X,C)

    # 4D cases
    if nd == 4:
        # (T,Y,X,C) or (Z,Y,X,C)
        if A.shape[-1] <= 8:
            return A  # already Z/T,Y,X,C — treat first dim as Z
        # (T,C,Y,X) or (Z,C,Y,X)
        if A.shape[1] <= 8:
            A2 = np.transpose(A, (0,2,3,1))  # -> (Z,Y,X,C)
            return A2
        # (C,Y,X,T) or (C,Y,X,Z)
        if A.shape[0] <= 8:
            A2 = np.transpose(A, (3,1,2,0))  # -> (Z,Y,X,C)
            return A2

    # Fallback: unable to infer reliably from the array alone.
    raise RuntimeError("Could not guess layout from array alone.")

def read_stack_ZYXC(path: str):
    """
    Read TIFF into (Z,Y,X,C) with **Z := T*Z** (frames folded into Z).
    Compatible with:
      - ImageJ hyperstacks (axes like 'TYXC', 'TCYX', etc.)
      - OME-TIFF axes
      - Your prealignment saves (pages=Z, samplesperpixel=C)
    Returns: (zyxc_float32, like_dtype)
    """
    with tiff.TiffFile(path) as tf:
        s = tf.series[0]
        arr = s.asarray()
        axes = getattr(s, "axes", "") or ""
        like_dtype = arr.dtype

        # Prefer OME/ImageJ axes if present
        if axes:
            try:
                zyxc = _from_axes(arr, axes)
                return zyxc.astype(np.float32, copy=False), like_dtype
            except Exception as e:
                print(f"[warn] axes='{axes}' failed on {os.path.basename(path)}: {e}")

        # Heuristic guess
        try:
            zyxc = _guess_layout(arr)
            return zyxc.astype(np.float32, copy=False), like_dtype
        except Exception:
            # Final fallback: treat each page as Z, with samples-per-pixel as C when present
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

    else:  # Mattes
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
    """centers_triplets: [(idx, middleZ_img_0..1, basename), ...], sorted by idx."""
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
    # Collect actual files present for the requested indices
    present = {}
    for i in indices:
        p = os.path.join(folder, f"{exp_id}_fish{fish}_s{i:02d}{input_suffix}")
        if os.path.exists(p):
            present[i] = p
        else:
            print(f"⚠️ missing: {p}")
    if not present:
        sys.exit("No stacks found.")

    # Damaged list (skip) - read basenames
    damaged_names = set()
    if os.path.exists(damaged_list_file):
        with open(damaged_list_file, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    damaged_names.add(os.path.basename(s))
        print(f"[damaged] loaded {len(damaged_names)} entries")
    else:
        print("[damaged] no damaged_stacks.txt found — proceeding with available stacks")

    # Build list of good indices (present and not damaged)
    good_indices = [i for i in sorted(present.keys()) if os.path.basename(present[i]) not in damaged_names]

    # Find longest consecutive run of good indices
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
        # finalize
        if cur_start is not None:
            if best_start is None or (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
        return (best_start, best_end) if best_start is not None else (None, None)

    run_start, run_end = longest_consecutive_run(good_indices)
    if run_start is None:
        sys.exit("No non-damaged consecutive run found to montage.")

    # Build all_paths only from that longest run
    all_paths = []
    for i in range(run_start, run_end + 1):
        p = present.get(i)
        if p is None:
            # gap - should not happen because we chose consecutive run, but guard
            print(f"[warn] expected present file for index {i} but missing")
            continue
        all_paths.append((i, p))

    print(f"[info] Selected longest non-damaged consecutive run: {run_start}..{run_end} -> {len(all_paths)} stacks")

    blocks = []            # aligned blocks, each (Z,Y,X,C) with Z := folded T/Z frames (Z := T*Z)
    centers_triplets = []  # (index, middle-Z image (0..1), basename)
    base_dtype = None

    # Sequential reference (prev good)
    prev_center = None    # 2D numpy (middle Z, REG_CHANNEL) from previous aligned block
    prev_ref    = None    # ANTs image of prev_center

    metric_tag = "CCfast" if REG_METRIC == "CC" else "MattesMI"

    for i, p in all_paths:
        name = os.path.basename(p)
        if name in damaged_names:
            print(f"[skip] damaged: {name}")
            continue

        print(f"\n[align] {name} (index {i})")
        vol, like_dtype = read_stack_ZYXC(p)  # <-- FULL BLOCK, (Z,Y,X,C)
        if base_dtype is None:
            base_dtype = like_dtype

        moving_center = center_slice_reg(vol).astype(np.float32)

        if prev_ref is None:
            # First good block: keep as-is (no registration), but apply per-ministack trimming
            first_block = vol.astype(np.float32)

            z0 = int(SKIP_Z_TOP) if SKIP_Z_TOP is not None else 0
            z1 = int(SKIP_Z_BOTTOM) if SKIP_Z_BOTTOM is not None else 0
            z0 = max(0, z0)
            z1 = max(0, z1)

            if (z0 + z1) >= first_block.shape[0]:
                print(f"[warn] first block would be empty after trimming: top={z0}, bottom={z1}, Z={first_block.shape[0]} — keeping untrimmed")
                trimmed_first = first_block
            else:
                trimmed_first = first_block[z0: first_block.shape[0] - z1] if z1 > 0 else first_block[z0:]

            blocks.append(trimmed_first)
            centers_triplets.append((i, to_float01(moving_center), name))
            prev_center = moving_center
            prev_ref    = ants.from_numpy(prev_center)
            print("[base] set as reference for subsequent stacks")
            continue

        # Register middle Z (REG_CHANNEL) to previous middle Z (REG_CHANNEL)
        if REG_METRIC == "CC":
            fwd, _ = register_2d(
                fixed_ref_np=prev_center, moving_np=moving_center, metric="CC",
                ds_factor=DS_FACTOR, cc_iters=CC_ITERS, aff_sampling=AFF_SAMPLING, use_masks=USE_MASKS
            )
        else:
            fwd, _ = register_2d(
                fixed_ref_np=prev_center, moving_np=moving_center, metric="Mattes", mi_iters=MI_ITERS
            )

        # Apply the rigid transform to ALL (Z,C) of this block
        aligned = ants_apply_rigid_2d_to_stack(vol, fwd, fixed2d_ref=prev_ref)

        # Append to master (optionally trim edge Z slices to hide seams)
        z0 = int(SKIP_Z_TOP) if SKIP_Z_TOP is not None else 0
        z1 = int(SKIP_Z_BOTTOM) if SKIP_Z_BOTTOM is not None else 0
        z0 = max(0, z0)
        z1 = max(0, z1)

        # Apply trimming per-ministack. Guard against over-trimming.
        if (z0 + z1) >= aligned.shape[0]:
            print(f"[warn] skipping block after trimming would remove all slices: top={z0}, bottom={z1}, Z={aligned.shape[0]}")
            continue

        trimmed = aligned[z0: aligned.shape[0] - z1] if z1 > 0 else aligned[z0:]
        blocks.append(trimmed)
        print(f"[concat] appended; trimmed Z {z0}..-{z1} -> block shape: {trimmed.shape}")

        # Update reference using aligned middle Z (REG_CHANNEL)
        aligned_center = center_slice_reg(aligned).astype(np.float32)
        centers_triplets.append((i, to_float01(aligned_center), name))
        prev_center = aligned_center
        prev_ref    = ants.from_numpy(prev_center)

    if not blocks:
        sys.exit("Nothing to concatenate (all stacks missing or damaged).")

    # Concatenate along Z (i.e., across all T frames of all substacks)
    master = np.concatenate(blocks, axis=0)  # (ΣZ, Y, X, C)
    print("\n[concat] final (Z,Y,X,C) shape:", master.shape)

    # Save outputs (Z as pages; C interleaved)
    slug = f"{exp_id}_fish{fish}"
    tag  = metric_tag

    run_tag = f"s{run_start:02d}-s{run_end:02d}"

    final_out = os.path.join(folder, f"{slug}_{run_tag}_montaged_{tag}.tif")
    write_bigtiff_zyxc(final_out, master, like_dtype=base_dtype)

    # Also export GCaMP channel alone (ch1) as a single-channel BigTIFF
    if master.shape[-1] >= 2:
        gcamp_out = os.path.join(folder, f"{slug}_{run_tag}_montaged_{tag}_GCaMP_ch1.tif")
        master_gcamp = master[:, :, :, 1:2]  # keep singleton channel dim -> (Z,Y,X,1)
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