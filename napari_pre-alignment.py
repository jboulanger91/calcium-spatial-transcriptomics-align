#!/usr/bin/env python
"""
napari_pre_alignment.py

Interactive napari tool to pre-align rostro–caudal confocal stacks by in-plane rotation and flips,
and to record a simple transform per stack (angle, vertical flip, left/right mirror, midline).

Key ideas
- Input stacks are multi-page TIFFs (often ImageJ hyperstacks).
- Data model: input series -> (Z, Y, X, C). If a singleton T exists, it is removed (or folded into Z).
- The user adjusts rotation + flips + a vertical midline guide using keyboard shortcuts.
- The same 2D transform (angle + flips about the midline) is applied to every Z slice and channel.
- Each pre-aligned stack is written as an ImageJ-compatible BigTIFF with axes "TZCYX" (T=1).
- A JSON file stores per-stack decisions, so sessions are resumable and updates are idempotent.

Folder layout (fixed by this script)
- Native inputs are read from:
  <fish_root>/native
- Prealigned outputs are written to:
  <fish_root>/prealigned

Output naming (CircuitSeeker-style)
Global identifiers:
    exp_id = "exp_001"
    fish   = 2
All outputs are named:
    {exp_id}_fish{fish}_s{section:02d}_pre.tif

The per-stack decision DB is stored at:
    <fish_root>/rotations_rc_{exp_id}_fish{fish}.json

Keyboard controls
- a / d     : -0.5° / +0.5°
- s / w     : -5°   / +5°
- Shift-f   : toggle vertical flip (top/bottom)
- x         : toggle left/right mirror about guide_x
- LEFT/RIGHT: move vertical guide line (±10 px)
- g         : auto angle from PCA mask (axis → vertical), then you can tweak
- r         : reset (angle=0, flips False, guide=center)
- n         : SAVE current stack and go to Next
- p         : go to Previous (no save unless you press n first)
- h         : print help to console
"""

from __future__ import annotations

import os
import json
from typing import List, Tuple

import numpy as np
import tifffile as tiff
from skimage import transform
from skimage.transform import AffineTransform, warp
import napari


# --------------------------- USER SETTINGS ---------------------------
# Global identifiers used for ALL output names
exp_id = "exp_001"   # e.g. "exp_001" or "exp1_110425"
fish = 2             # integer
run_id = f"{exp_id}_fish{fish}"

# Fish root folder (contains native/ and prealigned/)
fish_root = (
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/"
    "data/exp1_110425/oct_confocal_stacks/fish2"
)

native_dir = os.path.join(fish_root, "native")
prealigned_dir = os.path.join(fish_root, "prealigned")

# Input naming
prefix = "20x-4us-1um_DAPI_GFP488_RFP594_fish2-s1-"
indices = list(range(1, 25 + 1))  # 1..25

# Output naming
out_suffix = "_pre.tif"

# Persistent transform database (resumable)
# Expected full path example:
#   <fish_root>/<exp_id>_fish<fish>_rotations.json
#   e.g. exp_001_fish2_rotations.json
db_path = os.path.join(fish_root, f"{run_id}_rotations.json")
# --------------------------------------------------------------------


# ============================= I/O ===================================
def _drop_T_and_to_ZYXC(arr: np.ndarray, axes: str) -> np.ndarray:
    """
    Convert input array to (Z, Y, X, C).

    - If T exists:
        * If Z exists, folds T into Z (Z := Z*T).
        * Else, treats T as Z.
    - Ensures C and Z exist.
    - Ensures Y and X exist (if missing, assumes last two dims are YX).
    """
    A = arr
    axes = (axes or "").upper()

    # Fold or convert T
    if "T" in axes:
        tpos = axes.index("T")
        if "Z" in axes:
            zpos = axes.index("Z")
            # move T to just after Z, then merge into Z
            A = np.moveaxis(A, tpos, zpos + 1)
            axes = axes.replace("T", "")
            new_shape = list(A.shape)
            new_shape[zpos] *= new_shape[zpos + 1]
            del new_shape[zpos + 1]
            A = A.reshape(new_shape)
        else:
            # no Z axis: treat T as Z
            A = np.moveaxis(A, tpos, 0)
            axes = axes.replace("T", "Z")

    # Ensure C exists
    if "C" not in axes:
        A = A[..., None]
        axes += "C"

    # Ensure Z exists
    if "Z" not in axes:
        A = A[None, ...]
        axes = "Z" + axes

    # Ensure Y/X exist; if not, assume last two dims are YX
    if "Y" not in axes or "X" not in axes:
        while len(axes) < A.ndim:
            axes = "?" + axes
        axes = axes[:-2] + "YX"

    # Permute to Z,Y,X,C using a safe moveaxis pattern
    pos = {ax: i for i, ax in enumerate(axes)}
    src = [pos["Z"], pos["Y"], pos["X"], pos["C"]]
    dst = [0, 1, 2, 3]
    return np.moveaxis(A, src, dst)


def read_stack_zyxc(path: str) -> Tuple[np.ndarray, np.dtype]:
    """Read first TIFF series and return (Z,Y,X,C) plus the original dtype."""
    with tiff.TiffFile(path) as tf:
        s = tf.series[0]
        axes = getattr(s, "axes", "") or ""
        arr = s.asarray()
    vol = _drop_T_and_to_ZYXC(arr, axes)
    return vol, arr.dtype


# ============================ Utils ==================================
def ch0_center_slice(vol_zyxc: np.ndarray) -> np.ndarray:
    """Return middle-Z, channel 0 (2D)."""
    zc = vol_zyxc.shape[0] // 2
    return vol_zyxc[zc, :, :, 0]


def to_float01_for_view(x: np.ndarray) -> np.ndarray:
    """Display-only normalization for napari preview."""
    x = x.astype(np.float32, copy=False)
    if np.ptp(x) <= 0:
        mx = float(x.max()) if x.size else 1.0
        return np.clip(x / (mx + 1e-6), 0, 1)
    p1, p99 = np.percentile(x, (1, 99))
    if p99 <= p1:
        return np.clip((x - x.min()) / (x.max() - x.min() + 1e-6), 0, 1)
    return np.clip((x - p1) / (p99 - p1), 0, 1)


# --- PCA helpers (optional auto angle) ---
def make_mask(img2d: np.ndarray, sigma: float = 1.5, min_size: int = 3000) -> np.ndarray:
    x = to_float01_for_view(img2d)
    from skimage import morphology, filters, exposure

    x = exposure.equalize_adapthist(x, clip_limit=0.01)
    x = filters.gaussian(x, sigma=sigma, preserve_range=True)
    thr = filters.threshold_otsu(x)
    m = x > thr
    m = morphology.remove_small_objects(m, min_size=min_size)
    m = morphology.binary_closing(m, morphology.disk(5))
    m = morphology.binary_opening(m, morphology.disk(3))
    return m


def principal_axis_angle(img2d: np.ndarray) -> float:
    m = make_mask(img2d)
    ys, xs = np.nonzero(m)
    if xs.size < 100:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    ctr = pts.mean(axis=0)
    C = np.cov((pts - ctr).T)
    _, vecs = np.linalg.eigh(C)
    v = vecs[:, -1] / (np.linalg.norm(vecs[:, -1]) + 1e-12)
    theta = np.degrees(np.arctan2(-v[1], v[0]))
    if theta <= -90:
        theta += 180
    if theta > 90:
        theta -= 180
    return float(theta)


def _wrap180(a: float) -> float:
    return (a + 180) % 360 - 180


def angle_to_vertical(theta_deg: float) -> float:
    # rotate the principal axis to vertical (+90 or -90), pick the smaller magnitude
    r1 = _wrap180(90 - theta_deg)
    r2 = _wrap180(-90 - theta_deg)
    return r1 if abs(r1) <= abs(r2) else r2


def flip_lr_about_x(img: np.ndarray, guide_x: float) -> np.ndarray:
    """Mirror left/right about a vertical line at x=guide_x."""
    tform = AffineTransform(scale=(-1, 1), translation=(2 * guide_x, 0))
    out = warp(img, tform.inverse, order=1, preserve_range=True, mode="edge")
    return out.astype(img.dtype, copy=False)


def rotate_and_flip_volume_zyxc(
    vol: np.ndarray, angle_deg: float, flip_tb: bool, flip_lr: bool, guide_x: int
) -> np.ndarray:
    """Apply same in-plane rigid transform to all Z slices & channels. Returns float32."""
    Z, Y, X, C = vol.shape
    out = np.empty((Z, Y, X, C), dtype=np.float32)
    for z in range(Z):
        for c in range(C):
            im = vol[z, :, :, c]
            im2 = transform.rotate(
                im, angle=angle_deg, resize=False, preserve_range=True, order=1
            ).astype(np.float32, copy=False)
            if flip_tb:
                im2 = np.flip(im2, axis=0)
            if flip_lr:
                im2 = flip_lr_about_x(im2, guide_x)
            out[z, :, :, c] = im2
    return out


# =========================== App ===========================
class PreAlignApp:
    def __init__(self, in_dir: str, prefix: str, indices: List[int]):
        self.in_dir = in_dir
        self.prefix = prefix
        self.indices = indices

        # Build ordered list of (section_index, filepath)
        self.paths: List[Tuple[int, str]] = []
        for i in indices:
            p = os.path.join(in_dir, f"{prefix}{i}.tif")
            if os.path.exists(p):
                self.paths.append((i, p))
            else:
                print(f"⚠️ missing: {p}")
        if not self.paths:
            raise SystemExit("No stacks found.")

        # Load DB (resumable)
        self.db = {}
        if os.path.exists(db_path):
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    self.db = json.load(f)
                if not isinstance(self.db, dict):
                    print(f"[warn] DB at {db_path} is not a dict; ignoring.")
                    self.db = {}
                else:
                    print(f"[db] loaded {len(self.db)} entries from: {db_path}")
            except json.JSONDecodeError as e:
                print(f"[warn] Could not parse DB JSON at {db_path}: {e}")
                self.db = {}
        else:
            print(f"[db] no existing DB found (will create on save): {db_path}")

        # state
        self.i_idx = 0
        self.current_stack = None     # ZYXC
        self.current_dtype = None
        self.center_raw = None        # 2D center ch0
        self.angle = 0.0
        self.flip_tb = False
        self.flip_lr = False
        self.guide_x = None

        # napari viewer
        self.viewer = napari.Viewer()
        self.img_layer = None
        self.line_layer = None

        self._bind_keys()
        self._load_current()
        napari.run()

    def _db_key(self, section: int) -> str:
        """Stable key independent of raw filename."""
        return f"{run_id}_s{section:02d}"

    def _legacy_db_keys(self, section: int, path: str) -> List[str]:
        """Backward-compatible keys used by older versions of this tool."""
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        return [
            base,          # some versions keyed by basename
            stem,          # some versions keyed by stem
            str(section),  # some versions keyed by section number
        ]

    def _print_status(self):
        section, path = self.paths[self.i_idx]
        print(
            f"[status] {self.i_idx+1}/{len(self.paths)} section={section:02d} "
            f"file={os.path.basename(path)} "
            f"angle={self.angle:.2f}° flip_tb={self.flip_tb} flip_lr={self.flip_lr} guide_x={self.guide_x}"
        )

    # --- napari refresh ---
    def _preview_image(self) -> np.ndarray:
        zc = self.current_stack.shape[0] // 2
        im = self.current_stack[zc, :, :, 0]
        im2 = transform.rotate(im, angle=self.angle, resize=False, preserve_range=True, order=1).astype(
            np.float32, copy=False
        )
        if self.flip_tb:
            im2 = np.flip(im2, axis=0)
        if self.flip_lr:
            im2 = flip_lr_about_x(im2, self.guide_x)
        return im2

    def _refresh_layers(self):
        prev01 = to_float01_for_view(self._preview_image())

        name = os.path.basename(self.paths[self.i_idx][1])
        if self.img_layer is None:
            self.img_layer = self.viewer.add_image(prev01, name=name, rgb=False, colormap="gray")
        else:
            self.img_layer.data = prev01
            self.img_layer.name = name

        H, W = prev01.shape
        x = int(self.guide_x)
        line = np.array([[0, x], [H - 1, x]])
        if self.line_layer is None:
            self.line_layer = self.viewer.add_shapes(
                [line],
                shape_type="line",
                edge_color="yellow",
                edge_width=2,
                name="midline",
                blending="translucent",
                opacity=0.9,
            )
            self.line_layer.editable = False
            self.line_layer.selectable = False
            self.line_layer.interactive = False
        else:
            self.line_layer.data = [line]

    # --- load/save ---
    def _load_current(self):
        section, path = self.paths[self.i_idx]
        print(f"\n[load] section {section:02d}: {path}")

        vol_zyxc, like_dtype = read_stack_zyxc(path)

        if vol_zyxc.shape[-1] != 3:
            print(f"[warn] Expected C=3, got C={vol_zyxc.shape[-1]}")
        if vol_zyxc.shape[0] < 2:
            print(f"[warn] Expected Z>1, got Z={vol_zyxc.shape[0]}")

        self.current_stack = vol_zyxc
        self.current_dtype = like_dtype
        self.center_raw = ch0_center_slice(self.current_stack)

        H, W = self.center_raw.shape

        # Prefer the stable key, but fall back to older key schemes so existing JSONs still work.
        key = self._db_key(section)
        rec = self.db.get(key)
        used_key = key if rec is not None else None
        if rec is None:
            for k in self._legacy_db_keys(section, path):
                if k in self.db:
                    rec = self.db[k]
                    used_key = k
                    break

        if rec is None:
            rec = {"angle_deg": 0.0, "flip_tb": False, "flip_lr": False, "guide_x": W // 2}
        else:
            # If we loaded a legacy record, keep working under the stable key going forward.
            if used_key != key:
                print(f"[db] found legacy key '{used_key}' → using it to initialize stable key '{key}'")
                self.db[key] = rec

        self.angle = float(rec.get("angle_deg", 0.0))
        self.flip_tb = bool(rec.get("flip_tb", False))
        self.flip_lr = bool(rec.get("flip_lr", False))
        self.guide_x = int(rec.get("guide_x", W // 2))

        self._refresh_layers()
        self._print_status()

    def _save_current(self):
        section, _path = self.paths[self.i_idx]
        key = self._db_key(section)

        # persist decision
        self.db[key] = {
            "angle_deg": float(self.angle),
            "flip_tb": bool(self.flip_tb),
            "flip_lr": bool(self.flip_lr),
            "guide_x": int(self.guide_x),
        }
        os.makedirs(fish_root, exist_ok=True)
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(self.db, f, indent=2)

        # apply to whole stack
        vol_tx = rotate_and_flip_volume_zyxc(
            self.current_stack, self.angle, self.flip_tb, self.flip_lr, self.guide_x
        )

        # output name: exp_id_fishX_sYY_pre.tif
        os.makedirs(prealigned_dir, exist_ok=True)
        out_name = f"{run_id}_s{section:02d}{out_suffix}"
        out_path = os.path.join(prealigned_dir, out_name)

        # Write ImageJ-compatible hyperstack with explicit T and Z axes
        # vol_tx: (Z, Y, X, C) -> ImageJ wants (T, Z, C, Y, X) with axes="TZCYX"
        Z, Y, X, C = vol_tx.shape
        vol_ij = vol_tx.transpose(0, 3, 1, 2)  # (Z, C, Y, X)
        vol_ij = vol_ij[np.newaxis, ...]       # (T=1, Z, C, Y, X)

        tiff.imwrite(
            out_path,
            vol_ij.astype(self.current_dtype, copy=False),
            bigtiff=True,
            imagej=True,
            metadata={"axes": "TZCYX"},
        )

        print(f"[save] wrote: {out_path}")
        print(f"[db]   updated: {db_path}")

    # --- navigation ---
    def _next(self):
        if self.i_idx < len(self.paths) - 1:
            self.i_idx += 1
            self._load_current()
        else:
            print("✅ Reached last stack.")

    def _prev(self):
        if self.i_idx > 0:
            self.i_idx -= 1
            self._load_current()
        else:
            print("⛔ At first stack.")

    # --- auto guess ---
    def _auto_angle(self):
        theta = principal_axis_angle(self.center_raw)
        self.angle = float(angle_to_vertical(theta))
        print(f"[auto] PCA θ={theta:.2f}°, suggested angle={self.angle:.2f}°")
        self._refresh_layers()
        self._print_status()

    # --- key bindings ---
    def _bind_keys(self):
        @self.viewer.bind_key("a")
        def _dec_small(_viewer):  # noqa: N802
            self._bump_angle(-0.5)

        @self.viewer.bind_key("d")
        def _inc_small(_viewer):  # noqa: N802
            self._bump_angle(+0.5)

        @self.viewer.bind_key("s")
        def _dec_big(_viewer):  # noqa: N802
            self._bump_angle(-5.0)

        @self.viewer.bind_key("w")
        def _inc_big(_viewer):  # noqa: N802
            self._bump_angle(+5.0)

        @self.viewer.bind_key("Shift-F")
        def _toggle_tb(_viewer):  # noqa: N802
            self.flip_tb = not self.flip_tb
            self._refresh_layers()
            self._print_status()

        @self.viewer.bind_key("x")
        def _toggle_lr(_viewer):  # noqa: N802
            self.flip_lr = not self.flip_lr
            self._refresh_layers()
            self._print_status()

        @self.viewer.bind_key("left")
        def _guide_left(_viewer):  # noqa: N802
            self.guide_x = max(0, int(self.guide_x) - 10)
            self._refresh_layers()
            self._print_status()

        @self.viewer.bind_key("right")
        def _guide_right(_viewer):  # noqa: N802
            W = self.center_raw.shape[1]
            self.guide_x = min(W - 1, int(self.guide_x) + 10)
            self._refresh_layers()
            self._print_status()

        @self.viewer.bind_key("g")
        def _guess(_viewer):  # noqa: N802
            self._auto_angle()

        @self.viewer.bind_key("r")
        def _reset(_viewer):  # noqa: N802
            H, W = self.center_raw.shape
            self.angle, self.flip_tb, self.flip_lr, self.guide_x = 0.0, False, False, W // 2
            self._refresh_layers()
            self._print_status()

        @self.viewer.bind_key("n")
        def _save_next(_viewer):  # noqa: N802
            self._save_current()
            self._next()

        @self.viewer.bind_key("p")
        def _prev_no_save(_viewer):  # noqa: N802
            self._prev()

        @self.viewer.bind_key("h")
        def _help(_viewer):  # noqa: N802
            print(__doc__)
            self._print_status()

    def _bump_angle(self, delta: float):
        self.angle = _wrap180(self.angle + delta)
        self._refresh_layers()
        self._print_status()


# --------------------------- Entry point ---------------------------
if __name__ == "__main__":
    if not os.path.isdir(native_dir):
        raise SystemExit(f"Native input folder not found: {native_dir}")

    os.makedirs(prealigned_dir, exist_ok=True)
    print(f"[db] using DB path: {db_path}")

    PreAlignApp(in_dir=native_dir, prefix=prefix, indices=indices)