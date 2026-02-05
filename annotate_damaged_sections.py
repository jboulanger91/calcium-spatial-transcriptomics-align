"""
annotate_damaged_sections.py

Two-step workflow:
1) Interactive annotation (one stack at a time):
   - Mark each stack as damaged (d) or good (g).
   - If good, select the Z slice with the most signal (best_z) to use later for alignment.
   - Saves annotations to section_annotations.tsv

2) PDF QC report:
   - Shows thumbnails using:
       * best_z slice for good stacks (if annotated)
       * middle slice for damaged or unannotated stacks
   - Damaged stacks: semi-transparent red overlay
   - Longest consecutive stretch of NON-damaged: yellow border

Section order inferred from filenames like:
  ...fish2-s1-10.tif  -> section 10
If no section number can be parsed, files are ordered by name.

Inputs
- pre_rc_dir: folder containing *.tif stacks
- section_annotations.tsv (created/updated)
- damaged_stacks.txt (optional, maintained for backwards compatibility)

Outputs
- section_annotations.tsv saved inside pre_rc_dir
- damaged_stacks.txt updated inside pre_rc_dir
- damaged_sections_report.pdf saved inside pre_rc_dir
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tifffile as tif

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

import napari

# ---------------- user settings ----------------
pre_rc_dir = Path(
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/"
    "data/exp1_110425/oct_confocal_stacks/fish4/prealigned"
)

# Files inside pre_rc_dir
annotations_tsv = pre_rc_dir / "section_annotations.tsv"
damaged_list_file = pre_rc_dir / "damaged_stacks.txt"
out_pdf = pre_rc_dir / "damaged_sections_report.pdf"

# Display / PDF settings
channel_to_display = 0
thumbs_per_page = 12  # 3x4 grid
dpi = 200
vmin_percentile = 1
vmax_percentile = 99
# --------------------------------------------

SECTION_RE = re.compile(r"-s\d+-(\d+)(?:\D|$)")


@dataclass
class Annotation:
    path: str
    section: str
    damaged: int      # 1 damaged, 0 good
    best_z: str       # empty if damaged/unknown


# ---------------- filename parsing / ordering ----------------

def parse_section_number(p: Path) -> Optional[int]:
    m = SECTION_RE.search(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def sort_tifs(files: list[Path]) -> list[Path]:
    def sort_key(p: Path):
        sec = parse_section_number(p)
        return (0, sec) if sec is not None else (1, p.name)
    return sorted(files, key=sort_key)


# ---------------- I/O: annotations ----------------

def read_annotations(tsv_path: Path) -> Dict[str, Annotation]:
    out: Dict[str, Annotation] = {}
    if not tsv_path.exists():
        return out
    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            p = (row.get("path") or "").strip()
            if not p:
                continue
            out[p] = Annotation(
                path=p,
                section=(row.get("section") or "").strip(),
                damaged=int((row.get("damaged") or "0").strip() or 0),
                best_z=(row.get("best_z") or "").strip(),
            )
    return out


def write_annotations(tsv_path: Path, ann: Dict[str, Annotation]) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "section", "damaged", "best_z"], delimiter="\t"
        )
        writer.writeheader()
        for k in sorted(ann.keys()):
            a = ann[k]
            writer.writerow(
                {"path": a.path, "section": a.section, "damaged": a.damaged, "best_z": a.best_z}
            )


def sync_damaged_list(damaged_txt: Path, ann: Dict[str, Annotation]) -> None:
    damaged_txt.parent.mkdir(parents=True, exist_ok=True)
    damaged = [k for k, v in ann.items() if v.damaged == 1]
    with open(damaged_txt, "w") as f:
        for p in damaged:
            f.write(p + "\n")


# ---------------- image helpers ----------------

def robust_normalize(img: np.ndarray) -> np.ndarray:
    im = img.astype(np.float32, copy=False)
    lo, hi = np.percentile(im, [vmin_percentile, vmax_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(im)), float(np.max(im))
        if hi <= lo:
            return np.zeros_like(im, dtype=np.float32)
    im = (im - lo) / (hi - lo)
    return np.clip(im, 0.0, 1.0)


def _find_channel_axis(shape: Tuple[int, ...], max_channels: int = 8) -> Optional[int]:
    """
    Heuristic: channel axis is often small (<= max_channels) and stack has >=3 dims.
    Returns axis index or None.
    """
    candidates = [i for i, s in enumerate(shape) if 1 < s <= max_channels]
    if not candidates:
        return None
    # prefer axis that is not one of the two largest spatial dimensions
    sorted_axes = sorted(range(len(shape)), key=lambda i: shape[i], reverse=True)
    likely_xy = set(sorted_axes[:2])  # two largest dims
    for ax in candidates:
        if ax not in likely_xy:
            return ax
    # fallback: take smallest candidate
    return min(candidates, key=lambda i: shape[i])


def _reorder_3d_to_zyx(vol3: np.ndarray) -> np.ndarray:
    """
    Accept a 3D volume with unknown axis order and return (Z, Y, X)
    by treating the two largest axes as Y and X, remaining axis as Z.
    """
    if vol3.ndim != 3:
        raise ValueError(f"_reorder_3d_to_zyx expects 3D, got {vol3.ndim}D")

    shape = vol3.shape
    axes_sorted = sorted(range(3), key=lambda i: shape[i], reverse=True)
    y_ax, x_ax = axes_sorted[0], axes_sorted[1]
    z_ax = [a for a in range(3) if a not in (y_ax, x_ax)][0]

    # target order: (z, y, x)
    return np.moveaxis(vol3, (z_ax, y_ax, x_ax), (0, 1, 2))


def to_zyx_single_channel(stack: np.ndarray, channel: int) -> np.ndarray:
    """
    Convert stack to a 3D (Z, Y, X) volume for a single channel.

    Supported inputs:
      - 3D: any axis order (e.g. ZYX, YXZ, etc.)
      - 4D: with a channel axis somewhere (heuristic <= 8)

    If 4D but no obvious channel axis is found, raises ValueError.
    """
    if stack.ndim == 3:
        return _reorder_3d_to_zyx(stack)

    if stack.ndim == 4:
        ch_ax = _find_channel_axis(stack.shape, max_channels=8)
        if ch_ax is None:
            raise ValueError(f"4D stack but no channel axis found, shape={stack.shape}")

        n_ch = stack.shape[ch_ax]
        c = int(np.clip(channel, 0, n_ch - 1))
        vol3 = np.take(stack, indices=c, axis=ch_ax)  # now 3D
        return _reorder_3d_to_zyx(vol3)

    raise ValueError(f"Unsupported stack ndim={stack.ndim}, shape={stack.shape}")


def get_2d_from_zyx(vol_zyx: np.ndarray, z: int) -> np.ndarray:
    z = int(np.clip(z, 0, vol_zyx.shape[0] - 1))
    return vol_zyx[z]


# ---------------- napari annotation ----------------

def annotate_with_napari(pre_rc_dir: Path) -> None:
    pre_rc_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sort_tifs(list(pre_rc_dir.glob("*.tif")))
    if not tif_files:
        print(f"[annotate] No TIFF files found in: {pre_rc_dir}")
        return

    ann = read_annotations(annotations_tsv)
    idx = {"i": 0}
    current = {"path": None, "vol_zyx": None}

    viewer = napari.Viewer()

    help_text = (
        "\nNapari controls:\n"
        "  g = mark GOOD (not damaged)\n"
        "  d = mark DAMAGED\n"
        "  b = save current Z as best_z (for GOOD stacks)\n"
        "  n / Right = next stack\n"
        "  p / Left  = previous stack\n"
        "  q = save + quit\n"
    )
    print(help_text)

    def save_and_sync():
        write_annotations(annotations_tsv, ann)
        sync_damaged_list(damaged_list_file, ann)

    def set_window_title(i: int, p: Path):
        p_abs = str(p)
        sec = parse_section_number(p)
        sec_str = "" if sec is None else str(sec)
        a = ann.get(p_abs)
        status = "UNLABELED"
        if a is not None:
            status = "DAMAGED" if a.damaged == 1 else f"GOOD best_z={a.best_z or '??'}"
        viewer.window._qt_window.setWindowTitle(
            f"[{i+1}/{len(tif_files)}] {p.name}  section={sec_str or 'NA'}  {status}  [ch={channel_to_display}]"
        )

    def is_last_index(i: int) -> bool:
        return i >= (len(tif_files) - 1)

    def finish_and_close():
        """Save, print a message, and close the viewer if open."""
        save_and_sync()
        print(f"[annotate] Reached end: saved annotations to: {annotations_tsv}")
        print(f"[annotate] Synced damaged list to: {damaged_list_file}")
        try:
            viewer.close()
        except Exception:
            pass

    def load_i(i: int) -> None:
        i = int(np.clip(i, 0, len(tif_files) - 1))
        idx["i"] = i

        p = tif_files[i].resolve()
        stack = tif.imread(str(p))

        # Standardize to (Z,Y,X) for the selected channel
        vol_zyx = to_zyx_single_channel(stack, channel_to_display)
        if vol_zyx.ndim != 3 or vol_zyx.shape[0] < 1:
            raise ValueError(f"After standardization expected 3D with Z>=1, got {vol_zyx.shape}")

        current["path"] = p
        current["vol_zyx"] = vol_zyx

        viewer.layers.clear()
        viewer.add_image(
            vol_zyx,
            name=p.name,
            contrast_limits=None,
        )

        # Ensure napari is in 2D view with a Z slider (dims.ndisplay=2)
        viewer.dims.ndisplay = 2

        # Put Z at mid-slice initially (nice starting point)
        viewer.dims.set_current_step(0, vol_zyx.shape[0] // 2)

        set_window_title(i, p)

    def get_current_z() -> int:
        vol_zyx = current["vol_zyx"]
        if vol_zyx is None:
            return 0
        # After we standardize to (Z,Y,X), axis 0 is always Z
        z = int(viewer.dims.current_step[0])
        return int(np.clip(z, 0, vol_zyx.shape[0] - 1))

    @viewer.bind_key("g")
    def mark_good(v):
        p = current["path"]
        if p is None:
            return
        p_abs = str(p)
        sec = parse_section_number(p)
        existing = ann.get(p_abs)
        best_z = existing.best_z if existing is not None else ""
        ann[p_abs] = Annotation(
            path=p_abs,
            section="" if sec is None else str(sec),
            damaged=0,
            best_z=best_z,
        )
        save_and_sync()
        set_window_title(idx["i"], p)
        print(f"[annotate] {p.name}: marked GOOD")

        # If this was the last stack, finish
        if is_last_index(idx["i"]):
            finish_and_close()

    @viewer.bind_key("d")
    def mark_damaged(v):
        p = current["path"]
        if p is None:
            return
        p_abs = str(p)
        sec = parse_section_number(p)
        ann[p_abs] = Annotation(
            path=p_abs,
            section="" if sec is None else str(sec),
            damaged=1,
            best_z="",
        )
        save_and_sync()
        set_window_title(idx["i"], p)
        print(f"[annotate] {p.name}: marked DAMAGED")

        # If this was the last stack, finish
        if is_last_index(idx["i"]):
            finish_and_close()

    @viewer.bind_key("b")
    def set_best_z(v):
        p = current["path"]
        if p is None:
            return
        p_abs = str(p)
        a = ann.get(p_abs)
        if a is None or a.damaged == 1:
            print("[best_z] Mark as GOOD first (press 'g').")
            return
        z = get_current_z()
        a.best_z = str(z)
        ann[p_abs] = a
        save_and_sync()
        set_window_title(idx["i"], p)
        print(f"[best_z] saved z={z} for {p.name}")

        # If on last stack, finish
        if is_last_index(idx["i"]):
            finish_and_close()

    @viewer.bind_key("n")
    def next_stack(v):
        next_i = min(idx["i"] + 1, len(tif_files) - 1)
        # if already at last, finish
        if is_last_index(idx["i"]):
            finish_and_close()
            return
        load_i(next_i)
        # if we just moved to the last index, auto-finish on next 'n' press (user still gets to inspect)
        if is_last_index(next_i):
            print("[annotate] Reached last stack — press 'n' again to save & quit, or press 'q' to save & quit now.")

    @viewer.bind_key("Right")
    def next_stack_right(v):
        # same behavior as 'n'
        next_stack(v)

    @viewer.bind_key("p")
    def prev_stack(v):
        if idx["i"] <= 0:
            load_i(0)
        else:
            load_i(max(idx["i"] - 1, 0))

    @viewer.bind_key("Left")
    def prev_stack_left(v):
        prev_stack(v)

    @viewer.bind_key("q")
    def quit_and_save(v):
        finish_and_close()

    # start
    load_i(0)
    napari.run()  # blocks until you close (or finish_and_close closes viewer)

    # ensure last-save (in case viewer was closed unexpectedly)
    save_and_sync()
    print(f"[annotate] Exiting: annotations saved to: {annotations_tsv}")
    print(f"[annotate] Damaged list synced to: {damaged_list_file}")


# ---------------- report generation ----------------

def longest_non_damaged_run(items: list[tuple[Path, bool]]) -> tuple[int, int]:
    best_start = best_end = -1
    best_len = 0
    cur_start = None

    for i, (_, is_dmg) in enumerate(items):
        if not is_dmg:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                    best_end = i - 1
                cur_start = None

    if cur_start is not None:
        cur_len = len(items) - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
            best_end = len(items) - 1

    return best_start, best_end


def make_report(pre_rc_dir: Path) -> None:
    pre_rc_dir.mkdir(parents=True, exist_ok=True)

    ann = read_annotations(annotations_tsv)

    tif_files = sort_tifs(list(pre_rc_dir.glob("*.tif")))
    if not tif_files:
        print(f"[report] No TIFF files found in: {pre_rc_dir}")
        return

    # items: keep damaged flag for overlays + longest-run
    items: list[tuple[Path, bool]] = []
    for p in tif_files:
        p_abs = p.resolve()
        a = ann.get(str(p_abs))
        is_damaged = (a is not None and a.damaged == 1)
        items.append((p_abs, is_damaged))

    run_start, run_end = longest_non_damaged_run(items)

    nrows, ncols = 3, 4
    per_page = thumbs_per_page
    assert per_page == nrows * ncols, "thumbs_per_page must match grid (3x4 = 12)"

    with PdfPages(out_pdf) as pdf:
        for page_start in range(0, len(items), per_page):
            page_items = items[page_start: page_start + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5), dpi=dpi)
            axes = np.array(axes).reshape(-1)

            for ax_i, ax in enumerate(axes):
                ax.axis("off")
                if ax_i >= len(page_items):
                    continue

                path, is_dmg = page_items[ax_i]
                a = ann.get(str(path))

                try:
                    stack = tif.imread(str(path))
                    vol_zyx = to_zyx_single_channel(stack, channel_to_display)

                    # --- STRICT: show annotated best_z for GOOD stacks when available ---
                    used_z = None
                    bestz_label = ""

                    if a is not None and a.damaged == 0:
                        if a.best_z != "":
                            used_z = int(a.best_z)
                            bestz_label = f"[best_z={a.best_z}]"
                        else:
                            # Good but not annotated yet
                            used_z = vol_zyx.shape[0] // 2
                            bestz_label = "[best_z=NA]"
                    else:
                        # Damaged or unlabeled -> middle slice
                        used_z = vol_zyx.shape[0] // 2
                        # optionally label unlabeled
                        if a is None:
                            bestz_label = "[unlabeled]"
                        else:
                            bestz_label = ""  # damaged doesn't need best_z label

                    img2d = get_2d_from_zyx(vol_zyx, used_z)
                    disp = robust_normalize(img2d)

                except Exception as e:
                    ax.text(
                        0.5, 0.5, f"ERROR\n{path.name}\n{e}",
                        ha="center", va="center", fontsize=8, transform=ax.transAxes
                    )
                    continue

                ax.imshow(disp, cmap="gray", interpolation="nearest")

                name = path.name
                fish_idx = name.find("fish")
                title = name[fish_idx:] if fish_idx != -1 else name
                ax.set_title(f"{title}  {bestz_label}", fontsize=8)

                # Red overlay if damaged
                if is_dmg:
                    ax.add_patch(
                        Rectangle(
                            (0, 0), 1, 1,
                            transform=ax.transAxes,
                            facecolor="red",
                            alpha=0.22,
                            edgecolor="none",
                            zorder=10,
                        )
                    )

                # Yellow border for longest non-damaged run
                global_idx = page_start + ax_i
                in_best_run = (
                    run_start != -1
                    and run_start <= global_idx <= run_end
                    and (not is_dmg)
                )
                if in_best_run:
                    ax.add_patch(
                        Rectangle(
                            (0, 0), 1, 1,
                            transform=ax.transAxes,
                            fill=False,
                            edgecolor="yellow",
                            linewidth=4,
                            zorder=20,
                        )
                    )

            fig.suptitle(
                "Prealigned/pre-RC sections — damaged shaded red; longest OK run outlined yellow\n"
                f"Folder: {pre_rc_dir}\n"
                f"Channel: {channel_to_display} | Annotations: {annotations_tsv.name}",
                fontsize=10
            )
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[report] Wrote PDF: {out_pdf}")


# ---------------- CLI entrypoint ----------------

def main():
    pre_rc_dir.mkdir(parents=True, exist_ok=True)

    valid_modes = {"annotate", "report", "all"}

    # Jupyter often passes args like --f=...; pick the first valid mode anywhere
    mode = None
    for arg in sys.argv[1:]:
        a = arg.lower()
        if a in valid_modes:
            mode = a
            break
    if mode is None:
        mode = "annotate"

    print(f"[main] Mode: {mode}")

    if mode == "annotate":
        annotate_with_napari(pre_rc_dir)
    elif mode == "report":
        make_report(pre_rc_dir)
    elif mode == "all":
        annotate_with_napari(pre_rc_dir)
        make_report(pre_rc_dir)


if __name__ == "__main__":
    main()