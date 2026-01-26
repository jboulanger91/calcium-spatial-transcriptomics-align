import os
from pathlib import Path
import tifffile as tif
import matplotlib.pyplot as plt

# ---------------- user inputs ----------------
pre_rc_dir = Path(
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/calcium-spatial-transcriptomics-align/"
    "data/exp1_110425/oct_confocal_stacks/benchmark_data/fish2/prealigned_rc"
)

damaged_list_file = pre_rc_dir / "damaged_stacks.txt"

channel_to_display = 0     # e.g. DAPI
z_slice = None             # None = middle slice
# --------------------------------------------


def load_stack(path):
    stack = tif.imread(path)
    return stack


def get_display_slice(stack):
    """
    Expected shapes:
      (Z, Y, X)
      (Z, Y, X, C)
      (C, Z, Y, X)
    """
    if stack.ndim == 3:
        z = stack.shape[0] // 2 if z_slice is None else z_slice
        return stack[z]

    if stack.ndim == 4:
        # guess channel last
        if stack.shape[-1] <= 4:
            z = stack.shape[0] // 2 if z_slice is None else z_slice
            return stack[z, :, :, channel_to_display]
        else:
            # channel first
            z = stack.shape[1] // 2 if z_slice is None else z_slice
            return stack[channel_to_display, z]

    raise ValueError(f"Unsupported stack shape: {stack.shape}")


# create damaged_stacks.txt if needed
damaged_list_file.touch(exist_ok=True)

# load already-flagged stacks (avoid duplicates)
with open(damaged_list_file, "r") as f:
    already_flagged = set(line.strip() for line in f if line.strip())

tif_files = sorted(pre_rc_dir.glob("*.tif"))

print(f"Found {len(tif_files)} pre-aligned stacks")
print("Controls: [y] damaged | [n] ok | [q] quit\n")

for tif_path in tif_files:
    tif_path = tif_path.resolve()

    if str(tif_path) in already_flagged:
        continue

    print(f"Viewing: {tif_path.name}")

    stack = load_stack(tif_path)
    img = get_display_slice(stack)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(tif_path.name)
    plt.axis("off")
    plt.show(block=False)

    resp = input("Damaged? [y/n/q]: ").strip().lower()

    plt.close()

    if resp == "q":
        print("Stopping QC.")
        break

    if resp == "y":
        with open(damaged_list_file, "a") as f:
            f.write(str(tif_path) + "\n")
        print("→ marked as damaged\n")
    else:
        print("→ ok\n")

print("QC finished.")