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

# Multimodal volumetric stack registration (Napari + ANTs)

This repository implements a **robust, reproducible pipeline for registering large 3D microscopy volumes across modalities** (e.g. confocal ↔ functional imaging). The workflow combines lightweight **interactive pre-alignment**, automated **quality control and montage construction**, and **multi-stage ANTs registration** with explicit physical spacing and full parameter provenance.

The code is designed for real experimental data: anisotropic voxels, partial tissue loss, damaged sections, large global scale differences, and substantial non-linear deformations introduced by tissue processing.

---

## Overview of the workflow

**Input:** multi-channel 3D TIFF stacks (e.g. Olympus OIR converted to TIFF)

**Output:**
- fully registered NIfTI volumes (fixed ↔ moving)
- ImageJ-ready 2‑channel QC overlays
- timestamped JSON records capturing the *exact ANTs command used*

High-level steps:
1. **Interactive pre-alignment (Napari)** — enforce consistent orientation
2. **Visual QC & damaged-section detection** — identify usable blocks
3. **Montage construction** — build a clean reference volume
4. **ANTs registration** — Rigid → Similarity → Affine → SyN (non-linear)

---

## Repository contents

### `napari_pre-alignment.py`
Interactive Napari tool to quickly rotate and flip stacks so they share a common anatomical frame (rostro–caudal axis, left/right convention). A simple guide line and keyboard shortcuts allow fast correction of dozens of stacks.

### `annotate_damaged_sections.py`
Automated QC utility that:
- scans pre-aligned stacks
- generates a multi-page PDF showing a representative slice per stack
- highlights damaged sections
- identifies the **longest consecutive run of clean sections**

This step ensures that downstream montage and registration are performed only on high-quality data.

### `montage_register_prealigned.py`
Builds a clean reference volume from multiple adjacent stacks:
- loads pre-aligned stacks as `(Z, Y, X, C)`
- skips sections marked as damaged
- registers each stack to its neighbor using 2D rigid alignment on the central Z slice
- applies the transform to all slices/channels
- concatenates stacks into a single 3D montage

### `ANTs_register_without_mask.py`
Main registration driver based on **ANTs**:
- converts fixed and moving TIFF stacks to NIfTI with **explicit voxel spacing**
- runs a multi-stage ANTs pipeline:
  **Rigid → Similarity → Affine → SyN**
- supports large non-linear deformations (tissue expansion, bending)
- writes:
  - canonical warped volumes
  - ImageJ-compatible 2‑channel overlays
  - **timestamped JSON files containing the full `antsRegistration` command** for reproducibility

This script is intentionally explicit: all ANTs parameters are visible, versionable, and comparable across runs.

---

## Typical usage

### 0) Convert raw data to TIFF

Raw imaging data (e.g. Olympus `.oir`) should be batch-converted to TIFF using **Fiji / ImageJ**:

1. `Process → Batch → Convert…`
2. Input: folder with `.oir`
3. Output format: `TIFF`

Bio-Formats preserves channels, Z-planes, and bit depth.

---

### 1) Pre-align stacks (Napari)

```bash
python napari_pre-alignment.py
```

Interactively rotate/flip stacks so all volumes share a consistent orientation before automated processing.

---

### 2) QC and damaged-section detection

```bash
python annotate_damaged_sections.py
```

Produces a PDF report and a `damaged_stacks.txt` file used by the montage step.

---

### 3) Montage clean sections

```bash
python montage_register_prealigned.py
```

Builds a single, clean reference volume from the longest contiguous run of non-damaged stacks.

---

### 4) Multimodal registration with ANTs

```bash
python ANTs_register_without_mask.py \
  --fixed  /path/to/fixed_montage.tif \
  --moving /path/to/moving_stack.tif \
  --fixed-spacing-um  0.621 0.621 1.0 \
  --moving-spacing-um 0.396 0.396 2.0 \
  --exp-id exp_001 \
  --fish 2 \
  --out-dir /path/to/ANTs_output
```

Each run produces:
- `exp_001_fish2_warped.nii.gz` (latest result)
- `exp_001_fish2_fixed_warped_2ch.tif` (latest overlay)
- `exp_001_fish2_<timestamp>_overlay.tif`
- `exp_001_fish2_<timestamp>_ants_params.json`

The JSON file records the **full ANTs command**, input paths, spacings, and outputs, enabling exact reproduction of any run.

---

## Environment

A Conda environment file is provided:

```bash
conda env create -f stx-py310.yaml
conda activate stx-py310
```

This installs ANTs dependencies, Napari, scientific Python libraries, and Bio-Formats support.

---

## Design principles

- **Explicit physical units** — voxel spacing is always set deliberately
- **Coarse-to-fine alignment** — large capture range before refinement
- **Human-in-the-loop QC** — visual inspection is integrated, not an afterthought
- **Full provenance** — every registration run is logged and reproducible

This pipeline reflects best practices for large-scale multimodal volumetric alignment in real experimental settings.

---