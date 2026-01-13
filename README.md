# Multimodal stack registration (CircuitSeeker + ANTs)

This repo contains a small set of scripts/notebooks to **pre-align**, **montage**, and **register** volumetric microscopy stacks. Two registration routes are supported:

1. **CircuitSeeker** (Python pipeline; see the example notebook)
2. **ANTs / ANTsPy** (convert TIFF → NIfTI and run ANTs registration from Python)

A lightweight **Napari pre-alignment** step is also included to quickly rotate/flip stacks so that the rostro–caudal axis is consistent before running automated registration.

---

## Contents

- `napari_pre-alignment.py`  
  Interactive Napari tool to pre-align stacks (rotate/flip + guide line) and save corrected TIFFs.

- `align_many_from_center2D_CC-Mattes_with-prealignment.py`  
  Sequential “montage” alignment for a series of already pre-aligned stacks:
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

### 0) (Recommended) Pre-alignment in Napari
Use `napari_pre-alignment.py` to make sure all stacks share the same “up” direction and left/right convention (quick rotate + flip + midline guide).

This step is especially helpful if you have multiple tiles/blocks that were acquired with slightly different orientations.

---

## Option A — CircuitSeeker registration (notebook-driven)

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

## Option B — ANTs registration (script)

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