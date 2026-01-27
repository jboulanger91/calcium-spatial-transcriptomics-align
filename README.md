# Multimodal stack registration (CircuitSeeker + ANTs)

This repo contains a small set of scripts/notebooks to **pre-align**, **montage**, and **register** volumetric microscopy stacks. Two registration routes are supported:

1. **CircuitSeeker** (Python pipeline; see the example notebook)
2. **ANTs / ANTsPy** (convert TIFF → NIfTI and run ANTs registration from Python)

A lightweight **Napari pre-alignment** step is also included to quickly rotate/flip stacks so that the rostro–caudal axis is consistent before running automated registration.

---

## Contents

- `napari_pre-alignment.py`  
  Interactive Napari tool to pre-align stacks (rotate/flip + guide line) and save corrected TIFFs.

- `annotate_damaged_sections.py`  
  Generate a PDF QC report over pre-aligned stacks, highlighting damaged sections and the longest consecutive clean run.

- `montage_register_prealigned.py`  
  Sequential montage construction for a series of already pre-aligned stacks:
  - loads each stack as `(Z, Y, X, C)`
  - registers the **middle Z slice** (channel 0) of stack *i* to stack *i-1*
  - applies the same 2D rigid transform to **all slices/channels** of stack *i*
  - concatenates blocks into one final aligned volume + produces a QC PNG of aligned center slices

- `tif2nii_and_register.py`  
  ANTs-based registration driver:
  - reads TIFF stacks
  - writes NIfTI with **independent voxel spacings** for fixed and moving images
  - runs ANTs registration (via `antspyx`)
  - optionally keeps intermediate NIfTIs

- `multimodal_registration_example.ipynb`  
  CircuitSeeker-based multimodal alignment example (foreground detection, moments alignment, whole-image alignment, transform inversion, etc.). This notebook imports CircuitSeeker modules (alignment pipeline, transforms, axis alignment).

---

## Typical workflow

### Image format conversion (OIR → TIFF)

Raw imaging data were acquired in Olympus `.oir` format. Prior to any alignment or registration, all `.oir` files should be batch-converted to `.tiff` using **ImageJ/Fiji**, ensuring compatibility with downstream tools (Napari, CircuitSeeker, ANTs).

#### Batch conversion using Fiji (GUI)

1. Open **Fiji**
2. Navigate to **Process → Batch → Convert…**
3. Set:
   - **Input**: folder containing `.oir` files
   - **Output**: destination folder for `.tiff` files
   - **Output format**: `TIFF`
4. Click **OK** to start batch conversion

Fiji uses the **Bio-Formats** importer by default, preserving image metadata (bit depth, channels, Z-planes). Multi-channel or multi-plane datasets are saved as multi-page TIFFs.


### 0) (Recommended) Pre-alignment in Napari
Use `napari_pre-alignment.py` to make sure all stacks share the same “up” direction and left/right convention (quick rotate + flip + midline guide).

This step is especially helpful if you have multiple tiles/blocks that were acquired with slightly different orientations.

### 1) Visual QC of pre-aligned stacks (PDF report)

After pre-alignment, stacks are reviewed using a lightweight, non-interactive QC report to identify damaged or corrupted volumes (e.g. incomplete acquisition, severe motion, missing slices).

Use:

- `annotate_damaged_sections.py`

This script:
- scans all pre-aligned / pre-RC TIFF stacks in a folder
- generates a multi-page PDF showing a thumbnail (middle Z slice, selected channel) for each stack
- overlays **red shading** on stacks listed in `damaged_stacks.txt`
- automatically identifies and highlights the **longest consecutive stretch of non-damaged sections** (yellow outline)
- creates `damaged_stacks.txt` automatically if it does not exist

The output `damaged_sections_report.pdf` is saved alongside the stacks and provides a compact visual summary of data quality and usable section ranges.

### 2) Montage construction from clean sections

Use:

- `montage_register_prealigned.py`

This script:
- loads pre-aligned stacks as `(Z, Y, X, C)`
- parses section indices from filenames (e.g. `fish2-s1-10`)
- skips sections listed as damaged in `damaged_stacks.txt`
- automatically selects the **longest consecutive run of non-damaged sections**
- registers each stack to the previous one using 2D rigid alignment on the middle Z slice (channel 0)
- applies the resulting transform to all Z slices and channels
- concatenates aligned stacks into a single 3D volume
- optionally trims edge Z slices per stack to hide seams

This produces a single montaged volume and a QC image of aligned center slices.

---

## Option A — ANTs registration (script)

Use:

- `tif2nii_and_register.py`

High-level idea:
1. Convert fixed + moving TIFF stacks to NIfTI (`.nii.gz`) **with correct physical spacing**
2. Run ANTs registration using `antspyx`

Example:

```bash
python3 tif2nii_and_register.py \
  --fixed  /path/to/fixed.tif \
  --moving /path/to/moving.tif \
  --fixed-spacing-um  0.48 0.48 2.0 \
  --moving-spacing-um 0.32 0.32 1.0 \
  --out-prefix reg_ \
  --keep-nii
  ```

---

## Option B — ANTs registration (script)

Open and run:

- `multimodal_registration_example.ipynb`

This notebook demonstrates a CircuitSeeker pipeline including:
- basic preprocessing / “obvious corrections”
- foreground detection
- coarse alignment (moments / principal axes)
- whole-image alignment
- “wiggle” refinement
- inverting displacement fields / transforms

> CircuitSeeker must be installed and importable for this path.

---
