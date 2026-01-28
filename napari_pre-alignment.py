#!/usr/bin/env python
"""
napari_pre_alignment.py

Interactive napari tool to pre-align rostro–caudal confocal stacks by in-plane rotation and flips,
and to record a simple transform per stack (angle, vertical flip, left/right mirror, midline).

Key ideas
- Input stacks are already roughly oriented and stored as multi-page TIFFs.
- Data model: input series -> (Z, Y, X, C) with singleton T dropped; C is typically 3 channels.
- The user adjusts a vertical midline and rotation/flip flags using keyboard shortcuts.
- The same 2D transform (angle + flips about the midline) is applied to every Z slice and channel.
- Each pre-aligned stack is written as a BigTIFF with Z pages, each page shaped (Y, X, C).
- A JSON file stores per-stack decisions, so sessions are resumable and adjustments are idempotent.

Inputs
- A folder of raw stacks with names: <prefix><index>.tif
- Optional existing rotations_rc.json DB for reloading previous choices.

Outputs
- prealigned_rc/<basename>_preRC.tif : pre-aligned stacks
- rotations_rc.json                  : per-stack transform metadata

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

import os, json
from typing import List, Tuple
import numpy as np
import tifffile as tiff
from skimage import transform
from skimage.transform import AffineTransform, warp
import napari

# --------------------------- USER SETTINGS ---------------------------
folder     = "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/data/exp1_110425/oct_confocal_stacks/benchmark_data/fish2/native"
prefix     = "20x-4us-1um_DAPI_GFP488_RFP594_fish2-s1-"
indices    = list(range(1, 25+1))  # e.g. 1..25
    
 # Where to write pre-aligned stacks
out_dir    = os.path.join(os.path.abspath(folder), "prealigned")
out_suffix = "_pre.tif"

# Persistent transform database (per file basename)
db_path    = os.path.join(folder, "rotations_rc.json")
# --------------------------------------------------------------------


# ============================= I/O ===================================
def _drop_T_and_to_ZYXC(arr: np.ndarray, axes: str) -> np.ndarray:
    """Convert input to (Z,Y,X,C). Drop singleton T if present. No rescale."""
    A = arr
    axes = (axes or "").upper()

    # Drop T dimension entirely by folding it into Z (T is treated as additional Z slices)
    if 'T' in axes:
        tpos = axes.index('T')
        T = A.shape[tpos]
        # move T next to Z (or front if Z absent)
        if 'Z' in axes:
            zpos = axes.index('Z')
            A = np.moveaxis(A, tpos, zpos + 1)
            axes = axes.replace('T', '')
            # merge T into Z
            new_shape = list(A.shape)
            new_shape[zpos] *= new_shape[zpos + 1]
            del new_shape[zpos + 1]
            A = A.reshape(new_shape)
            axes = axes.replace('Z', 'Z')
        else:
            # no Z axis: treat T as Z
            A = np.moveaxis(A, tpos, 0)
            axes = axes.replace('T', 'Z')

    # Ensure C exists
    if 'C' not in axes:
        A = A[..., None]
        axes += 'C'

    # Ensure Z exists
    if 'Z' not in axes:
        A = A[None, ...]
        axes = 'Z' + axes

    # Ensure Y,X exist; if not, assume last two are Y,X
    if 'Y' not in axes or 'X' not in axes:
        # pad labels to match ndim
        while len(axes) < A.ndim:
            axes = '?' + axes
        # force last two are Y,X
        axes = axes[:-2] + 'YX'

    # Permute to Z,Y,X,C
    pos = {ax:i for i,ax in enumerate(axes)}
    order = [pos['Z'], pos['Y'], pos['X'], pos['C']]
    zyxc = np.moveaxis(A, order, [0,1,2,3])
    return zyxc


def read_stack_zyxc(path: str) -> Tuple[np.ndarray, np.dtype]:
    """Read first series → (Z,Y,X,C) with C=3; preserve dtype (no rescale)."""
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
    p1, p99 = np.percentile(x, (1,99)) if np.ptp(x) > 0 else (x.min(), x.max() or 1.0)
    if p99 > p1:
        x = (x - p1) / (p99 - p1)
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return np.clip(x, 0, 1)

# --- PCA helpers (optional auto angle) ---
def make_mask(img2d: np.ndarray, sigma=1.5, min_size=3000):
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
    if theta <= -90: theta += 180
    if theta >   90: theta -= 180
    return float(theta)

def wrap180(a: float) -> float:
    return ((a + 180) % 360)

def angle_to_vertical(theta_deg: float) -> float:
    r1 = wrap180( 90 - theta_deg)
    r2 = wrap180(-90 - theta_deg)
    return r1 if abs(r1) <= abs(r2) else r2

# --- flips/rotation ---
def flip_lr_about_x(img: np.ndarray, guide_x: float) -> np.ndarray:
    t = AffineTransform(scale=(-1, 1), translation=(2*guide_x, 0))
    out = warp(img, t.inverse, order=1, preserve_range=True, mode='edge')
    return out.astype(img.dtype, copy=False)

def rotate_and_flip_volume_zyxc(vol: np.ndarray, angle_deg: float,
                                flip_tb: bool, flip_lr: bool, guide_x: int) -> np.ndarray:
    """Apply same in-plane rigid to all Z & C. Output float32; preserve_range=True."""
    Z, Y, X, C = vol.shape
    out = np.empty_like(vol, dtype=np.float32)
    for z in range(Z):
        for c in range(C):
            im = vol[z, :, :, c]
            im2 = transform.rotate(im, angle=angle_deg, resize=False,
                                   preserve_range=True, order=1).astype(np.float32, copy=False)
            if flip_tb:
                im2 = np.flip(im2, axis=0)
            if flip_lr:
                im2 = flip_lr_about_x(im2, guide_x)
            out[z, :, :, c] = im2
    return out
# =====================================================================


# =========================== App ===========================
class PreAlignApp:
    def __init__(self, folder: str, prefix: str, indices: List[int]):
        self.folder = folder
        self.prefix = prefix
        self.indices = indices

        # build ordered list
        self.paths: List[Tuple[int, str]] = []
        for i in indices:
            p = os.path.join(folder, f"{prefix}{i}.tif")
            if os.path.exists(p):
                self.paths.append((i, p))
            else:
                print(f"⚠️ missing: {p}")
        if not self.paths:
            raise SystemExit("No stacks found.")

        # DB
        if os.path.exists(db_path):
            with open(db_path, "r") as f:
                self.db = json.load(f)
        else:
            self.db = {}

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

    # --- helpers ---
    def _basename(self, path: str) -> str:
        return os.path.basename(path)

    def _db_key(self, path: str) -> str:
        return self._basename(path)

    def _print_status(self):
        idx, path = self.paths[self.i_idx]
        base = self._basename(path)
        print(f"[status] {idx}/{self.paths[-1][0]} file={base} "
              f"angle={self.angle:.2f}° flip_tb={self.flip_tb} flip_lr={self.flip_lr} guide_x={self.guide_x}")

    # --- napari refresh ---
    def _preview_image(self) -> np.ndarray:
        zc = self.current_stack.shape[0] // 2
        im = self.current_stack[zc, :, :, 0]
        im2 = transform.rotate(im, angle=self.angle, resize=False,
                               preserve_range=True, order=1).astype(np.float32, copy=False)
        if self.flip_tb:
            im2 = np.flip(im2, axis=0)
        if self.flip_lr:
            im2 = flip_lr_about_x(im2, self.guide_x)
        return im2

    def _refresh_layers(self):
        prev = self._preview_image()
        prev01 = to_float01_for_view(prev)

        if self.img_layer is None:
            self.img_layer = self.viewer.add_image(prev01, name=self._basename(self.paths[self.i_idx][1]),
                                                   rgb=False, colormap='gray')
        else:
            self.img_layer.data = prev01
            self.img_layer.name = self._basename(self.paths[self.i_idx][1])

        H, W = prev01.shape
        x = self.guide_x
        line = np.array([[0, x], [H-1, x]])
        if self.line_layer is None:
            self.line_layer = self.viewer.add_shapes([line],
                                                     shape_type='line',
                                                     edge_color='yellow',
                                                     edge_width=2,
                                                     name='midline',
                                                     blending='translucent',
                                                     opacity=0.9)
            self.line_layer.editable = False
            self.line_layer.selectable = False
            self.line_layer.interactive = False
        else:
            self.line_layer.data = [line]

    # --- load/save ---
    def _load_current(self):
        idx, path = self.paths[self.i_idx]
        print(f"\n[load] {idx}: {path}")
        vol_zyxc, like_dtype = read_stack_zyxc(path)

        if vol_zyxc.shape[-1] != 3:
            print(f"[warn] Expected C=3, got C={vol_zyxc.shape[-1]}")
        if vol_zyxc.shape[0] < 2:
            print(f"[warn] Expected Z>1, got Z={vol_zyxc.shape[0]}")

        self.current_stack = vol_zyxc
        self.current_dtype = like_dtype
        self.center_raw = ch0_center_slice(self.current_stack)

        key = self._db_key(path)
        H, W = self.center_raw.shape
        rec = self.db.get(key, {"angle_deg": 0.0, "flip_tb": False, "flip_lr": False, "guide_x": W//2})
        self.angle   = float(rec.get("angle_deg", 0.0))
        self.flip_tb = bool(rec.get("flip_tb", False))
        self.flip_lr = bool(rec.get("flip_lr", False))
        self.guide_x = int(rec.get("guide_x", W//2))

        self._refresh_layers()
        self._print_status()

    def _save_current(self):
        idx, path = self.paths[self.i_idx]
        base = self._basename(path)
        key  = self._db_key(path)

        # persist decision
        self.db[key] = {
            "angle_deg": float(self.angle),
            "flip_tb": bool(self.flip_tb),
            "flip_lr": bool(self.flip_lr),
            "guide_x": int(self.guide_x),
        }
        with open(db_path, "w") as f:
            json.dump(self.db, f, indent=2)

        # apply to whole stack
        vol_tx = rotate_and_flip_volume_zyxc(self.current_stack, self.angle, self.flip_tb, self.flip_lr, self.guide_x)

        # write Z pages (Y,X,C) inline (no separate helper)
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.splitext(base)[0] + out_suffix
        out_path = os.path.join(out_dir, out_name)

        # Write ImageJ-compatible hyperstack with explicit T and Z axes
        # Input vol_tx has shape (Z, Y, X, C)
        # ImageJ requires (T, Z, C, Y, X) with axes="TZCYX"
        Z, Y, X, C = vol_tx.shape
        vol_ij = vol_tx.transpose(0, 3, 1, 2)   # (Z, C, Y, X)
        vol_ij = vol_ij[np.newaxis, ...]        # (T=1, Z, C, Y, X)

        tiff.imwrite(
            out_path,
            vol_ij.astype(self.current_dtype, copy=False),
            bigtiff=True,
            imagej=True,
            metadata={"axes": "TZCYX"}
        )

        print(f"[save] wrote: {out_path}")

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
        rot_to_vertical = angle_to_vertical(theta)
        self.angle = float(rot_to_vertical)
        print(f"[auto] PCA θ={theta:.2f}°, suggested angle={self.angle:.2f}°")
        self._refresh_layers()
        self._print_status()

    # --- key bindings ---
    def _bind_keys(self):
        @self.viewer.bind_key('a')
        def _dec_small(viewer): self._bump_angle(-0.5)

        @self.viewer.bind_key('d')
        def _inc_small(viewer): self._bump_angle(+0.5)

        @self.viewer.bind_key('s')
        def _dec_big(viewer): self._bump_angle(-5.0)

        @self.viewer.bind_key('w')
        def _inc_big(viewer): self._bump_angle(+5.0)

        @self.viewer.bind_key('Shift-F')
        def _toggle_tb(viewer):
            self.flip_tb = not self.flip_tb
            self._refresh_layers(); self._print_status()

        @self.viewer.bind_key('x')
        def _toggle_lr(viewer):
            self.flip_lr = not self.flip_lr
            self._refresh_layers(); self._print_status()

        @self.viewer.bind_key('left')
        def _guide_left(viewer):
            self.guide_x = max(0, self.guide_x - 10)
            self._refresh_layers(); self._print_status()

        @self.viewer.bind_key('right')
        def _guide_right(viewer):
            W = self.center_raw.shape[1]
            self.guide_x = min(W-1, self.guide_x + 10)
            self._refresh_layers(); self._print_status()

        @self.viewer.bind_key('g')
        def _guess(viewer): self._auto_angle()

        @self.viewer.bind_key('r')
        def _reset(viewer):
            H, W = self.center_raw.shape
            self.angle, self.flip_tb, self.flip_lr, self.guide_x = 0.0, False, False, W//2
            self._refresh_layers(); self._print_status()

        @self.viewer.bind_key('n')
        def _save_next(viewer):
            self._save_current()
            self._next()

        @self.viewer.bind_key('p')
        def _prev_no_save(viewer): self._prev()

        @self.viewer.bind_key('h')
        def _help(viewer):
            print(__doc__)
            self._print_status()

    def _bump_angle(self, delta: float):
        self.angle += delta
        if self.angle <= -180: self.angle += 360
        if self.angle >   180: self.angle -= 360
        self._refresh_layers(); self._print_status()


# --------------------------- Entry point ---------------------------
if __name__ == "__main__":
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")
    os.makedirs(out_dir, exist_ok=True)
    PreAlignApp(folder=folder, prefix=prefix, indices=indices)