#!/usr/bin/env python
"""
Sequential montage of pre-oriented TIFF stacks with 2D center-slice rigid alignment (T treated as Z).

- Input stacks are already prealigned & saved with `write_bigtiff_zyxc`, i.e. pages = frames, C interleaved.
- Reads each stack as (Z,Y,X,C), registering the middle Z (ch 0) to the previous non-damaged block.
- Applies that single 2D rigid transform to ALL (Z,C) slices of the block.
- Concatenates full aligned blocks (optionally dropping 1st Z to hide seams).
- Saves:
    * {prefix_sanitized}_FINAL_concat_{metricTag}.tif
    * {prefix_sanitized}_aligned_centers_{metricTag}.png  (QC: aligned middle-Z per block)

No flips, no user interaction. Damaged stacks are skipped.
"""

import os, sys, re
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import ants  # antspyx

# ===================== USER SETTINGS =====================
folder        = "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/spatial_transcriptomics/exp1_110425/oct_confocal_stacks/fish4_tifs/prealigned_rc"
prefix        = "20x-4us-1um_DAPI_GFP488_RFP594_fish4-s"
input_suffix  = "_preRC.tif"          # files like: <prefix><index>_preRC.tif
indices       = list(range(1, 30))    # 1..29

damaged_list_file = os.path.join(folder, "damaged_stacks.txt")
skip_dup_first    = True              # drop first Z of each appended block (except first)

# Registration choice: "CC" (fast) or "Mattes" (mutual information)
REG_METRIC = "Mattes"                 # "CC" or "Mattes"
# CC params
DS_FACTOR     = 4
CC_ITERS      = (300, 150, 75)
AFF_SAMPLING  = 256
USE_MASKS     = True
# Mattes params
MI_ITERS      = (1000, 500, 250)

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

def center_slice_Z0(vol_ZYXC):
    zmid = vol_ZYXC.shape[0] // 2
    return vol_ZYXC[zmid, :, :, 0]

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
    """Write (Z,Y,X,C) BigTIFF; no rescale; cast/clip to like_dtype."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Z, Y, X, C = arr_zyxc.shape
    with tiff.TiffWriter(path, bigtiff=True) as tw:
        for z in range(Z):
            page = arr_zyxc[z]
            if np.issubdtype(page.dtype, np.floating) and np.issubdtype(like_dtype, np.integer):
                info = np.iinfo(like_dtype)
                page = np.clip(page, info.min, info.max).astype(like_dtype)
            else:
                page = page.astype(like_dtype, copy=False)
            tw.write(page[..., 0] if C == 1 else page)
# ============================================


# ============================= Robust reader (T -> Z) =============================
def _permute(a: np.ndarray, src: list[int], dst: list[int]) -> np.ndarray:
    """np.moveaxis but for ordered groups."""
    return np.moveaxis(a, src, dst) if src != dst else a

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

    # Fallback: treat pages as Z, grayscale
    with tiff.TiffFile(BytesIO()) as _:
        pass  # just to keep imports happy if editor complains

    # Build Z from pages
    try:
        with tiff.TiffFile(BytesIO()) as _:
            pass
    except Exception:
        pass
    # Simpler fallback using pages:
    # (Re-open file is cleaner, but we only have the array here; defer to outer fallback.)
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
    # Ordered inputs: ...-s1_preRC.tif, ...-s2_preRC.tif, ...
    all_paths = []
    for i in indices:
        p = os.path.join(folder, f"{prefix}{i}{input_suffix}")
        if os.path.exists(p):
            all_paths.append((i, p))
        else:
            print(f"⚠️ missing: {p}")
    if not all_paths:
        sys.exit("No stacks found.")

    # Damaged list (skip)
    damaged_names = set()
    if os.path.exists(damaged_list_file):
        with open(damaged_list_file, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    damaged_names.add(os.path.basename(s))
        print(f"[damaged] loaded {len(damaged_names)} entries")
    else:
        print("[damaged] no damaged_stacks.txt found — proceeding with all available stacks")

    blocks = []            # aligned blocks, each (Z,Y,X,C) with Z := original T (or T*Z)
    centers_triplets = []  # (index, middle-Z image (0..1), basename)
    base_dtype = None

    # Sequential reference (prev good)
    prev_center = None    # 2D numpy (middle Z, ch0) from previous aligned block
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

        moving_center = center_slice_Z0(vol).astype(np.float32)

        if prev_ref is None:
            # First good block: keep as-is (no registration)
            blocks.append(vol.astype(np.float32))
            centers_triplets.append((i, to_float01(moving_center), name))
            prev_center = moving_center
            prev_ref    = ants.from_numpy(prev_center)
            print("[base] set as reference for subsequent stacks")
            continue

        # Register middle Z (ch0) to previous middle Z (ch0)
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

        # Append to master (optionally drop first Z for seam)
        if skip_dup_first and aligned.shape[0] > 1 and len(blocks) > 0:
            blocks.append(aligned[1:])
            print(f"[concat] appended (skipped first Z); block shape: {aligned[1:].shape}")
        else:
            blocks.append(aligned)
            print(f"[concat] appended; block shape: {aligned.shape}")

        # Update reference using aligned middle Z (ch0)
        aligned_center = center_slice_Z0(aligned).astype(np.float32)
        centers_triplets.append((i, to_float01(aligned_center), name))
        prev_center = aligned_center
        prev_ref    = ants.from_numpy(prev_center)

    if not blocks:
        sys.exit("Nothing to concatenate (all stacks missing or damaged).")

    # Concatenate along Z (i.e., across all T frames of all substacks)
    master = np.concatenate(blocks, axis=0)  # (ΣZ, Y, X, C)
    print("\n[concat] final (Z,Y,X,C) shape:", master.shape)

    # Save outputs (Z as pages; C interleaved)
    slug = sanitize_prefix(prefix)
    tag  = metric_tag

    final_out = os.path.join(folder, f"{slug}_FINAL_concat_{tag}.tif")
    write_bigtiff_zyxc(final_out, master, like_dtype=base_dtype)

    centers_png = os.path.join(folder, f"{slug}_aligned_centers_{tag}.png")
    save_centers_figure(centers_triplets, centers_png, cols=FIG_COLS, dpi=FIG_DPI)

    print("[save]")
    print(" ", final_out)
    print(" ", centers_png)
    print("✅ Done.")