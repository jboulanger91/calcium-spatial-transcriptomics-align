"""
annotate_damaged_sections.py

Generate a PDF QC report for a folder of pre-aligned (pre-RC) confocal mini-stacks.

For each stack, the script displays a thumbnail (middle Z slice, selected channel) and annotates:
- Damaged stacks: semi-transparent red overlay (paths listed in damaged_stacks.txt)
- Longest consecutive stretch of non-damaged sections: yellow border

Section order is inferred from filenames containing patterns like:
  ...fish2-s1-10.tif  -> section 10
If no section number can be parsed, files are ordered by name.

Inputs
- pre_rc_dir: folder containing *.tif stacks
- damaged_stacks.txt (optional): one path per line (absolute or relative). Created empty if missing.

Outputs
- damaged_sections_report.pdf saved inside pre_rc_dir

No user interaction.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# ---------------- user settings ----------------
pre_rc_dir = Path(
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/"
    "data/exp1_110425/oct_confocal_stacks/benchmark_data/fish2/prealigned"
)

# List of damaged stacks (one absolute path per line)
damaged_list_file = pre_rc_dir / "damaged_stacks.txt"

# Output PDF report
out_pdf = pre_rc_dir / "damaged_sections_report.pdf"

# Display settings
channel_to_display = 0     # e.g. DAPI if channel-last or channel-first stacks
z_slice = None             # None = middle slice
thumbs_per_page = 12       # 12 = 3x4 grid
dpi = 200
# --------------------------------------------


SECTION_RE = re.compile(r"-s\d+-(\d+)(?:\D|$)")


def parse_section_number(p: Path) -> int | None:
    """
    Try to parse a section index from filenames like:
    ...fish2-s1-10.tif  -> 10
    Returns None if not found.
    """
    m = SECTION_RE.search(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_stack(path: Path) -> np.ndarray:
    return tif.imread(str(path))


def get_display_slice(stack: np.ndarray) -> np.ndarray:
    """
    Expected shapes:
      (Z, Y, X)
      (Z, Y, X, C)  [channel last]
      (C, Z, Y, X)  [channel first]
    Returns a 2D array (Y, X).
    """
    if stack.ndim == 3:
        z = stack.shape[0] // 2 if z_slice is None else int(z_slice)
        return stack[z]

    if stack.ndim == 4:
        # heuristic: if last dim is small, it's probably channels
        if stack.shape[-1] <= 8:
            z = stack.shape[0] // 2 if z_slice is None else int(z_slice)
            return stack[z, :, :, int(channel_to_display)]
        else:
            z = stack.shape[1] // 2 if z_slice is None else int(z_slice)
            return stack[int(channel_to_display), z]

    raise ValueError(f"Unsupported stack shape: {stack.shape}")


def robust_normalize(img: np.ndarray) -> np.ndarray:
    """
    Make thumbnails easier to visually compare without changing underlying data.
    Percentile normalization to [0,1].
    """
    im = img.astype(np.float32, copy=False)
    lo, hi = np.percentile(im, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(im)), float(np.max(im))
        if hi <= lo:
            return np.zeros_like(im, dtype=np.float32)
    im = (im - lo) / (hi - lo)
    return np.clip(im, 0.0, 1.0)


def longest_non_damaged_run(items: list[tuple[Path, bool]]) -> tuple[int, int]:
    """
    items: list of (path, is_damaged) in the intended sequential order.
    Returns (start_idx, end_idx) inclusive for the longest consecutive run of NON-damaged.
    If there are ties, returns the first.
    If all damaged, returns (-1, -1).
    """
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

    # handle run to end
    if cur_start is not None:
        cur_len = len(items) - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
            best_end = len(items) - 1

    return best_start, best_end


def main() -> None:
    """Write a multi-page PDF report to `out_pdf` for stacks in `pre_rc_dir`."""
    pre_rc_dir.mkdir(parents=True, exist_ok=True)

    # Create damaged list file if missing
    damaged_list_file.touch(exist_ok=True)

    # Read damaged paths (stored as strings, typically absolute paths)
    with open(damaged_list_file, "r") as f:
        damaged_paths = {line.strip() for line in f if line.strip()}

    tif_files = sorted(pre_rc_dir.glob("*.tif"))

    # Sort by parsed section number when available, else by name
    def sort_key(p: Path):
        sec = parse_section_number(p)
        return (0, sec) if sec is not None else (1, p.name)

    tif_files = sorted(tif_files, key=sort_key)

    items: list[tuple[Path, bool]] = []
    for p in tif_files:
        p_abs = p.resolve()
        is_damaged = str(p_abs) in damaged_paths or str(p) in damaged_paths
        items.append((p_abs, is_damaged))

    if not items:
        print(f"[report] No TIFF files found in: {pre_rc_dir}")
        return

    run_start, run_end = longest_non_damaged_run(items)
    if run_start != -1:
        print(
            f"[report] Longest non-damaged run: indices {run_start}..{run_end} "
            f"({run_end - run_start + 1} sections)"
        )
    else:
        print("[report] All sections are marked damaged (no non-damaged run found).")

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
                try:
                    stack = load_stack(path)
                    img2d = get_display_slice(stack)
                    disp = robust_normalize(img2d)
                except Exception as e:
                    # show error tile
                    ax.text(
                        0.5, 0.5, f"ERROR\n{path.name}\n{e}",
                        ha="center", va="center", fontsize=8, transform=ax.transAxes
                    )
                    continue

                ax.imshow(disp, cmap="gray", interpolation="nearest")
                # show concise title starting from 'fish'
                name = path.name
                fish_idx = name.find("fish")
                title = name[fish_idx:] if fish_idx != -1 else name
                ax.set_title(title, fontsize=8)

                # Red shading overlay if damaged
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

                # Yellow border for longest non-damaged stretch
                global_idx = page_start + ax_i
                in_best_run = (run_start != -1) and (run_start <= global_idx <= run_end) and (not is_dmg)
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
                f"Prealigned/pre-RC sections — damaged shaded red; longest OK run outlined yellow\n"
                f"Folder: {pre_rc_dir}",
                fontsize=10
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[report] Wrote PDF: {out_pdf}")


if __name__ == "__main__":
    main()