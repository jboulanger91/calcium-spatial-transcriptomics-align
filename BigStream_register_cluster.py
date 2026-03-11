#!/usr/bin/env python3
"""
BigStream rigid+affine global alignment + 3-pass piecewise refinement (optional deform in pass 2).

Conventions:
- arrays are (Z, Y, X)
- spacing vectors are (Z, Y, X) in microns

This script expects two versions of each stack:
- RAW stacks: used for mask generation + global rigid/affine alignment
- CONVOLUTED stacks: used ONLY to learn the piecewise (blockwise) refinement

Outputs:
- ImageJ-compatible 2-channel overlays (global QC + final piecewise QC)
- JSON metadata for reproducibility (alignment_metadata.json)
"""

from pathlib import Path
import argparse
import os
import json
from datetime import datetime

import numpy as np
import tifffile
from tifffile import imwrite as tiff_imwrite

from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing, binary_dilation
from skimage.measure import label

from bigstream.align import alignment_pipeline
import bigstream.transform as bst
from bigstream.piecewise_align import distributed_piecewise_alignment_pipeline
from bigstream.piecewise_transform import distributed_apply_transform
import copy


# =========================
# Paths & tags
# =========================


# Run timestamp (shared across all outputs from this invocation)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Cluster-friendly defaults (can be overridden by CLI args)
SUBMIT_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", ".")).resolve()
DATADIR = Path(os.environ.get("DATADIR", str(SUBMIT_DIR / "data"))).resolve()
OUT_PATH = Path(os.environ.get("OUTDIR", str(SUBMIT_DIR / "Bigstream_output"))).resolve()
WORKDIR = os.environ.get("WORKDIR", "")

# Default filenames (match your SLURM script)
DEFAULT_FIXED_NAME = "exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP.tif"
DEFAULT_MOVING_NAME = (
    "2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz_CARE.tif"
)

DEFAULT_FIXED_NAME_CONVOLUTED = "exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_gf50.tif"
DEFAULT_MOVING_NAME_CONVOLUTED = (
    "2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz_CARE_gf40.tif"
)

# ---- Identifiers ----
exp_id, fish, section = "exp_001", 2, 5-9
DEFAULT_RUN_ID = f"{exp_id}_fish{fish}_section{section}"

# ---- Spacing (Z, Y, X) in microns ----
FIXED_SPACING = np.array([1.0, 0.621, 0.621], dtype=float)

# NOTE: if your moving stack has coarse Z (e.g. 2 µm), set MOVING_BASE = [2.0, 0.396, 0.396]
# With CARE, the resolution is now isotropic
MOVING_BASE = np.array([0.396, 0.396, 0.396], dtype=float)

# scale moving spacing (apply to Y,X only)
SCALE_YX = 1.48
MOVING_SPACING = MOVING_BASE.copy()
MOVING_SPACING[1:] *= SCALE_YX  # scales Y and X only


# =========================
# Helpers
# =========================


# Runtime-bound output context for save_path(tag, ext)
CURRENT_OUT_DIR: Path | None = None
CURRENT_RUN_ID: str | None = None


def save_path(tag: str, ext: str) -> Path:
    """
    Format:
    <run_id>_<YYYYMMDD_HHMMSS>_<tag>.<ext>
    Example:
    exp_001_fish2_20260218_132325_moving.tif

    Uses CURRENT_OUT_DIR / CURRENT_RUN_ID set inside main().
    Falls back to OUT_PATH / DEFAULT_RUN_ID if not set.
    """
    out_dir = Path(CURRENT_OUT_DIR) if CURRENT_OUT_DIR is not None else OUT_PATH
    run_id = CURRENT_RUN_ID if CURRENT_RUN_ID is not None else DEFAULT_RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{run_id}_{RUN_TIMESTAMP}_{tag}.{ext}"


def load_tiff_float32(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.memmap(str(path)))
    return arr.astype(np.float32, copy=False)


def write_json(tag: str, payload: dict) -> Path:
    """Write a JSON file using save_path() with robust numpy/tuple sanitization."""
    out = save_path(tag, "json")
    with open(out, "w") as f:
        json.dump(sanitize_for_json(payload), f, indent=2)
    print(" saved:", out)
    return out


def print_paths(fixed_path: Path, moving_path: Path, fixed_conv_path: Path, moving_conv_path: Path, out_dir: Path, run_id: str):
    print("[paths]")
    print(" submit:", SUBMIT_DIR)
    print(" data  :", DATADIR)
    print(" work  :", WORKDIR)
    print(" fixed :", fixed_path)
    print(" moving:", moving_path)
    print(" fixed (conv):", fixed_conv_path)
    print(" moving(conv):", moving_conv_path)
    print(" note : raw used for masks+global+final resample; convoluted used ONLY for piecewise")
    print(" out   :", out_dir)
    print(" run_id:", run_id)


def _to_uint16_for_imagej(vol_zyx: np.ndarray) -> np.ndarray:
    if np.issubdtype(vol_zyx.dtype, np.floating):
        p1, p99 = np.percentile(vol_zyx, (0.1, 99.9))
        v = (vol_zyx - p1) / (p99 - p1) if p99 > p1 else np.clip(vol_zyx, 0, 1)
        return (np.clip(v, 0, 1) * 65535).astype(np.uint16)
    return np.clip(vol_zyx, 0, 65535).astype(np.uint16)


def homogenize_then_threshold_mask(
    vol,
    thresh=0.20,
    sigma_bg=25,
    sigma_smooth=1.5,
    close_shape=(5, 5, 5),
    dilate_iter=32,
    keep_largest=True,
):
    v = vol.astype(np.float32, copy=False)

    p1, p99 = np.percentile(v, (1, 99))
    v = np.clip((v - p1) / (p99 - p1 + 1e-6), 0, 1)

    bg = gaussian_filter(v, sigma=sigma_bg)
    v_corr = np.clip(v / (bg + 1e-6), 0, np.percentile(v, 99.9))

    p1c, p99c = np.percentile(v_corr, (1, 99))
    v_corr = np.clip((v_corr - p1c) / (p99c - p1c + 1e-6), 0, 1)

    v_corr = gaussian_filter(v_corr, sigma=sigma_smooth)
    m = v_corr > float(thresh)

    if keep_largest:
        lbl = label(m)
        if lbl.max() > 0:
            counts = np.bincount(lbl.ravel()); counts[0] = 0
            m = (lbl == counts.argmax())

    m = binary_fill_holes(m)
    m = binary_closing(m, structure=np.ones(close_shape, bool))
    m = binary_dilation(m, iterations=int(dilate_iter))

    return m.astype(np.uint8), v_corr


def parse_args():
    p = argparse.ArgumentParser(description="BigStream rigid+affine + 2-channel ImageJ overlay")
    p.add_argument("--fixed", type=Path, default=None, help="Path to fixed TIFF (prefer staged WORKDIR/fixed.tif)")
    p.add_argument("--moving", type=Path, default=None, help="Path to moving TIFF (prefer staged WORKDIR/moving.tif)")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--run-id", type=str, default=None, help="Run id prefix for outputs")
    return p.parse_args()


def resolve_paths(args):
    # Prefer CLI args; otherwise use staged WORKDIR if provided; otherwise DATADIR defaults.
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = OUT_PATH

    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = DEFAULT_RUN_ID

    if args.fixed is not None:
        fixed = args.fixed
    else:
        fixed = (Path(WORKDIR) / "fixed.tif") if WORKDIR else (DATADIR / DEFAULT_FIXED_NAME)

    if args.moving is not None:
        moving = args.moving
    else:
        moving = (Path(WORKDIR) / "moving.tif") if WORKDIR else (DATADIR / DEFAULT_MOVING_NAME)

    return fixed, moving, out_dir, run_id


def sanitize_for_json(obj):
    """Convert numpy / tuples into JSON-safe structures."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    return obj




# =========================
# Alignment parameter blocks
# =========================

GLOBAL_STEPS = [
    ("rigid", dict(
        metric="MMI",
        metric_args={"numberOfHistogramBins": 32},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=1.0, minStep=1e-4, numberOfIterations=600),
        sampling="RANDOM",
        sampling_percentage=0.20,
        shrink_factors=(32, 16, 8, 4, 2, 1),
        smooth_sigmas=(24.0, 12.0, 6.0, 3.0, 1.0, 0.0),
        initial_condition="CENTER",
    )),
    ("random", dict(
        random_iterations=8000,     # more candidates, but still cheap at coarse levels
        nreturn=5,                  # keep top 5 for the next step

        # “residual” search, not global
        max_translation=20.0,      # tune: ~2–4× expected residual (in your physical units)
        max_rotation=0.25,          # ~14°
        max_scale=1.03,             # ±2% (small but useful)
        max_shear=0.03,             # small shear

        alignment_spacing=8.0,      # coarse scoring is stable + fast
        metric="MMI",
        metric_args={"numberOfHistogramBins": 24},  # fewer bins = more robust for exploration

        sampling="RANDOM",
        sampling_percentage=0.10,   # keep cheap; random is just proposing seeds

        shrink_factors=(16, 8, 4),
        smooth_sigmas=(6.0, 3.0, 1.5),
    )),
    ("affine", dict(
        metric="MMI",
        metric_args={"numberOfHistogramBins": 48},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.12, minStep=5e-7, numberOfIterations=1200),
        sampling="RANDOM",
        sampling_percentage=0.20,
        shrink_factors=(16, 8, 4, 2, 1),
        smooth_sigmas=(6.0, 3.0, 1.5, 0.5, 0.0),
    )),
]

# ============================================================
# PASS 1 (coarse / global+local): large blocks
# ============================================================

PIECEWISE_STEPS_PASS1_LARGE = [
    ("random", dict(
        random_iterations=6000, nreturn=5,
        max_translation=22.0, max_rotation=0.07,
        max_scale=1.008, max_shear=0.01,
        alignment_spacing=6.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        sampling="RANDOM", sampling_percentage=0.4,
        shrink_factors=(2, 1), smooth_sigmas=(1.8, 0.0),
    )),
    ("affine", dict(
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.02, minStep=5e-7, numberOfIterations=200),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(4, 2, 1), smooth_sigmas=(2.0, 1.0, 0.0),
        initial_condition="IDENTITY",
    )),
]

BLOCKSIZE_PASS1 = (320, 320, 320)
OVERLAP_PASS1  = 0.35


# ============================================================
# PASS 2 (fine residuals): smaller blocks + residual search
# ============================================================

PIECEWISE_STEPS_PASS2_FINE = [
    ("random", dict(
        random_iterations=4000, nreturn=3,
        max_translation=14.0, max_rotation=0.045,
        max_scale=1.008, max_shear=0.008,
        alignment_spacing=2.5,
        metric="MMI", metric_args={"numberOfHistogramBins": 24},
        sampling="RANDOM", sampling_percentage=0.4,
        shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0),
    )),
    ("affine", dict(
        metric="ANC", metric_args={"radius": 6},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.04, minStep=2e-7, numberOfIterations=200),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0),
        initial_condition="IDENTITY",
    )),
]

BLOCKSIZE_PASS2 = (160, 160, 160)
OVERLAP_PASS2  = 0.50


# ============================================================
# PASS 3 (extra fine): tiny blocks, NO deform (fast + less mess)
#   - residual-only random + rigid to snap edges and clean seams
# ============================================================

PIECEWISE_STEPS_PASS3_TINY = [
    ("random", dict(
        random_iterations=2000, nreturn=1,
        max_translation=3.0, max_rotation=0.01,
        max_scale=None, max_shear=None,
        alignment_spacing=2.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 24},
        sampling="RANDOM", sampling_percentage=0.55,
        shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0),
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 5},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.02, minStep=2e-7, numberOfIterations=140),
        sampling="RANDOM", sampling_percentage=0.50,
        shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0),
        initial_condition="IDENTITY",
    )),
]

BLOCKSIZE_PASS3 = (80, 80, 80)
OVERLAP_PASS3  = 0.50


# Local cluster (single-node) defaults
CLUSTER_KWARGS_DEFAULT = {
    "cluster_type": "local_cluster",
    "n_workers": 8,
    "threads_per_worker": 4,
}

def main():
    args = parse_args()
    fixed_path, moving_path, out_dir, run_id = resolve_paths(args)

    global CURRENT_OUT_DIR, CURRENT_RUN_ID
    CURRENT_OUT_DIR = out_dir
    CURRENT_RUN_ID = run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convoluted paths (prefer WORKDIR staging if present)
    fixed_conv_path = (Path(WORKDIR) / "fixed_convoluted.tif") if WORKDIR else (DATADIR / DEFAULT_FIXED_NAME_CONVOLUTED)
    moving_conv_path = (Path(WORKDIR) / "moving_convoluted.tif") if WORKDIR else (DATADIR / DEFAULT_MOVING_NAME_CONVOLUTED)

    print_paths(fixed_path, moving_path, fixed_conv_path, moving_conv_path, out_dir, run_id)

    print("\n[spacing Z,Y,X]")
    print(" fixed :", FIXED_SPACING)
    print(" moving base:", MOVING_BASE)
    print(f" scale YX: {SCALE_YX}")
    print(" moving scaled:", MOVING_SPACING)

    print("\n[load original tiffs]")
    fix = load_tiff_float32(fixed_path)
    mov = load_tiff_float32(moving_path)
    print(" fixed :", fix.shape, fix.dtype)
    print(" moving:", mov.shape, mov.dtype)

    print("\n[load convoluted tiffs]")
    fix_convoluted = load_tiff_float32(fixed_conv_path)
    mov_convoluted = load_tiff_float32(moving_conv_path)
    print(" fixed_convoluted :", fix_convoluted.shape, fix_convoluted.dtype)
    print(" moving_convoluted:", mov_convoluted.shape, mov_convoluted.dtype)

    print("\n[masks]")
    fix_mask, fix_corr = homogenize_then_threshold_mask(fix, thresh=0.18, sigma_bg=40, dilate_iter=12)
    mov_mask, mov_corr = homogenize_then_threshold_mask(mov, thresh=0.30, sigma_bg=30, dilate_iter=24)
    print(" fix_mask fg voxels:", int(fix_mask.sum()))
    print(" mov_mask fg voxels:", int(mov_mask.sum()))

    print("\n[align]")
    tforms = alignment_pipeline(
        fix=fix, mov=mov,
        fix_spacing=FIXED_SPACING, mov_spacing=MOVING_SPACING,
        fix_mask=fix_mask, mov_mask=mov_mask,
        steps=GLOBAL_STEPS,
        return_format="independent",
        context="TEST",
    )
    print(" transforms returned:", len(tforms))

    composed = bst.compose_transform_list(tforms, FIXED_SPACING)
    global_aligned = bst.apply_transform(
        fix_convoluted, mov_convoluted,
        FIXED_SPACING, MOVING_SPACING,
        transform_list=[composed],
    )
    print(" aligned:", global_aligned.shape, global_aligned.dtype)

    # -----------------------------
    # Save overlay after GLOBAL alignment (CONVOLUTED stacks)
    # - Global transform is learned on RAW.
    # - For easier QC, we apply it to the CONVOLUTED moving stack and overlay
    #   against the CONVOLUTED fixed stack.
    # -----------------------------
    print("\n[save overlay: global (convoluted)]")

    global_aligned_convoluted = bst.apply_transform(
        fix_convoluted,
        mov_convoluted,
        FIXED_SPACING,
        MOVING_SPACING,
        transform_list=[composed],
    )

    fixed_u16 = _to_uint16_for_imagej(fix_convoluted)
    warped_u16 = _to_uint16_for_imagej(np.asarray(global_aligned_convoluted))

    zmin = min(fixed_u16.shape[0], warped_u16.shape[0])
    stacked_global = np.stack([fixed_u16[:zmin], warped_u16[:zmin]], axis=1)
    stacked_global = stacked_global[np.newaxis, ...]

    overlay_global_out = save_path("overlay_fixedConvolved_vs_globalAlignedConvolved", "tif")
    print(" writing:", overlay_global_out)
    tiff_imwrite(
        str(overlay_global_out),
        stacked_global,
        bigtiff=True,
        imagej=True,
        metadata={"axes": "TZCYX"},
    )

    # =============================
    # PASS 1 — COARSE (LARGE BLOCKS)
    # =============================


    # -----------------------------
    # LOCAL CLUSTER (single node)
    # -----------------------------
    CLUSTER_KWARGS = dict(CLUSTER_KWARGS_DEFAULT)

    tmpdir = (out_dir / ".tmp_bigstream").resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Save metadata (both passes)
    # -----------------------------
    meta = {
        "run_id": run_id,
        "timestamp": RUN_TIMESTAMP,
        "fixed_path": str(fixed_path),
        "moving_path": str(moving_path),
        "fixed_spacing_zyx_um": sanitize_for_json(FIXED_SPACING),
        "moving_spacing_zyx_um": sanitize_for_json(MOVING_SPACING),
        "mask_calls": {
            "fixed": {"thresh": 0.18, "sigma_bg": 40, "dilate_iter": 12},
            "moving": {"thresh": 0.30, "sigma_bg": 30, "dilate_iter": 24},
        },
        "global_steps": sanitize_for_json(GLOBAL_STEPS),
        "piecewise": {
            "pass1": {
                "steps": sanitize_for_json(PIECEWISE_STEPS_PASS1_LARGE),
                "blocksize_zyx": sanitize_for_json(BLOCKSIZE_PASS1),
                "overlap": OVERLAP_PASS1,
            },
            "pass2": {
                "steps": sanitize_for_json(PIECEWISE_STEPS_PASS2_FINE),
                "blocksize_zyx": sanitize_for_json(BLOCKSIZE_PASS2),
                "overlap": OVERLAP_PASS2,
            },
            "pass3": {
                "steps": sanitize_for_json(PIECEWISE_STEPS_PASS3_TINY),
                "blocksize_zyx": sanitize_for_json(BLOCKSIZE_PASS3),
                "overlap": OVERLAP_PASS3,
            },
        },
        "cluster_kwargs": sanitize_for_json(CLUSTER_KWARGS),
        "temporary_directory": str(tmpdir),
        "rebalance_for_missing_neighbors": True,
    }
    write_json("alignment_metadata", meta)

    # -----------------------------
    # GLOBAL AFFINE
    # -----------------------------
    global_affine = composed
    assert isinstance(global_affine, np.ndarray) and global_affine.shape == (4, 4)

    # -----------------------------
    # PASS 1: COARSE piecewise
    # Learn the local refinement on the CONVOLUTED images,
    # but keep masks + global affine coming from the RAW pipeline.
    # -----------------------------

    # Cluster check 
    from dask.distributed import get_client

    try:
        client = get_client()
    except ValueError:
        # cluster will be created by first distributed call
        pass

    local_transform_stage1 = distributed_piecewise_alignment_pipeline(
        fix=fix_convoluted,
        mov=mov_convoluted,
        fix_spacing=FIXED_SPACING,
        mov_spacing=MOVING_SPACING,
        steps=PIECEWISE_STEPS_PASS1_LARGE,
        blocksize=BLOCKSIZE_PASS1,
        overlap=OVERLAP_PASS1,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        static_transform_list=[global_affine],
        rebalance_for_missing_neighbors=True,
        cluster_kwargs=copy.deepcopy(CLUSTER_KWARGS),
    )

    # -----------------------------
    # PASS 2: FINE piecewise (CONVOLUTED)
    # IMPORTANT: local_transform_stage1 is already ONE transform (vector field).
    # -----------------------------
    local_transform_stage2 = distributed_piecewise_alignment_pipeline(
        fix=fix_convoluted,
        mov=mov_convoluted,
        fix_spacing=FIXED_SPACING,
        mov_spacing=MOVING_SPACING,
        steps=PIECEWISE_STEPS_PASS2_FINE,
        blocksize=BLOCKSIZE_PASS2,
        overlap=OVERLAP_PASS2,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        static_transform_list=[global_affine, local_transform_stage1],
        rebalance_for_missing_neighbors=True,
        cluster_kwargs=copy.deepcopy(CLUSTER_KWARGS),
    )

    # -----------------------------
    # PASS 3: EXTRA FINE piecewise (CONVOLUTED), NO deform
    # IMPORTANT: local_transform_stage{1,2} are each ONE transform (vector field).
    # -----------------------------
    local_transform_stage3 = distributed_piecewise_alignment_pipeline(
        fix=fix_convoluted,
        mov=mov_convoluted,
        fix_spacing=FIXED_SPACING,
        mov_spacing=MOVING_SPACING,
        steps=PIECEWISE_STEPS_PASS3_TINY,
        blocksize=BLOCKSIZE_PASS3,
        overlap=OVERLAP_PASS3,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        static_transform_list=[global_affine, local_transform_stage1, local_transform_stage2],
        rebalance_for_missing_neighbors=True,
        cluster_kwargs=copy.deepcopy(CLUSTER_KWARGS),
    )

    # Cluster check (now guaranteed to exist)
    try:
        client = get_client()
        info = client.scheduler_info()
        print("Workers:", len(info["workers"]))
        print("Threads per worker:", client.nthreads())
    except ValueError:
        print("No active Dask client detected.")

    # -----------------------------
    # APPLY transforms
    # - Apply to RAW stacks (for final scientific output)
    # - Also apply to CONVOLUTED stacks (QC / debugging)
    # -----------------------------

    # QC output on CONVOLUTED data
    local_aligned_convoluted = distributed_apply_transform(
        fix_zarr=fix_convoluted,
        mov_zarr=mov_convoluted,
        fix_spacing=FIXED_SPACING,
        mov_spacing=MOVING_SPACING,
        transform_list=[global_affine, local_transform_stage1, local_transform_stage2, local_transform_stage3],
        blocksize=BLOCKSIZE_PASS3,
        overlap=OVERLAP_PASS3,
        temporary_directory=str(tmpdir),
        cluster_kwargs=copy.deepcopy(CLUSTER_KWARGS),
    )

    # -----------------------------
    # Save ImageJ overlays
    # -----------------------------
    print("\n[save overlays]")

    # QC overlay on convoluted stacks
    fixedc_u16 = _to_uint16_for_imagej(fix_convoluted)
    warpedc_u16 = _to_uint16_for_imagej(np.asarray(local_aligned_convoluted))
    zminc = min(fixedc_u16.shape[0], warpedc_u16.shape[0])
    stackedc = np.stack([fixedc_u16[:zminc], warpedc_u16[:zminc]], axis=1)
    stackedc = stackedc[np.newaxis, ...]

    overlayc_out = save_path("overlay_fixedConvolved_vs_piecewise3passConvolved", "tif")
    print(" writing:", overlayc_out)
    tiff_imwrite(
        str(overlayc_out),
        stackedc,
        bigtiff=True,
        imagej=True,
        metadata={"axes": "TZCYX"},
    )

    print(" done.")


if __name__ == "__main__":
    main()