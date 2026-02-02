# Multimodal volumetric stack registration (Napari + ANTs)

This repository implements a pipeline for registering zebrafish two-photon (2P) functional calcium imaging volumes to immuno-DAPI stained OCT-embedded cryosection stacks. 

---

## Overview of the workflow

**Input:** multi-channel 3D TIFF stacks from immunostained OCT-embedded cryosection (from Olympus OIR converted to TIFF)

**Output:**
- registered NIfTI volumes (fixed ↔ moving)
- ImageJ-ready 2‑channel QC overlays
- timestamped JSON records capturing the *exact ANTs command used*

Pipeline steps:
1. **Interactive pre-alignment (Napari)** — enforce consistent orientation
2. **Visual QC & damaged-section detection** — identify usable sub-stacks
3. **Montage construction** — build a clean fixed volume
4. **ANTs registration** — Rigid → Similarity → SyN

---

## Repository contents

### `napari_pre-alignment.py`
Interactive Napari tool to quickly rotate and flip OCT sub-stacks along rostro–caudal axis. 

### `annotate_damaged_sections.py`
Automated QC utility that:
- scans pre-aligned OCT sub-stacks to identify the longest consecutive run of non-damaged sections
- generates a PDF showing a representative slice per sub-stack

### `montage_register_prealigned.py`
Builds a clean reference volume from multiple adjacent stacks:
- registers each stack to its neighbor using 2D rigid alignment on the central Z slice
- concatenates stacks into a single 3D montage

![Aligned sections after montage registration](aligned_sections.png)
*Example output of the montage step, showing multiple adjacent sections rigidly aligned prior to ANTs registration.*

### `ANTs_register_without_mask.py`
Main registration driver based on **ANTs**:
- converts fixed and moving TIFF stacks to NIfTI with **explicit voxel spacing**
- runs a multi-stage ANTs pipeline:
  **Rigid → Similarity → SyN**
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


### 2) QC and damaged-section detection

```bash
python3 annotate_damaged_sections.py
```

Produces a PDF report and a `damaged_stacks.txt` file used by the montage step.


### 3) Montage clean sections

```bash
python3 montage_register_prealigned.py
```

Builds a single, clean reference volume from the longest contiguous run of non-damaged stacks.


### 4) Multimodal registration with ANTs

```bash
python3 ANTs_register_without_mask.py \
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