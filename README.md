# Multimodal volumetric stack registration (Napari + ANTs)

This repository implements a **robust, reproducible pipeline for registering large 3D microscopy volumes across modalities** (e.g. confocal ↔ functional imaging). The workflow combines lightweight **interactive pre-alignment**, automated **quality control and montage construction**, and **multi-stage ANTs registration**.

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

![Aligned sections after montage registration](calcium-spatial-transcriptomics-align/aligned_sections.png)
*Example output of the montage step, showing multiple adjacent sections rigidly aligned and concatenated into a clean reference volume prior to ANTs registration.*

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
- `expX_fishY_warped.nii.gz` (latest result)
- `expX_fishY_fixed_warped_2ch.tif` (latest overlay)
- `expX_fishY_<timestamp>_overlay.tif`
- `expX_fishY_<timestamp>_ants_params.json`

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