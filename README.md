# Multimodal volumetric stack registration

![Aligned sections after montage registration](aligned_sections.png)

---

## Overview

**Input:** multi-channel 3D TIFF stacks (e.g. Olympus OIR → TIFF)

**Output:**
- registered volumes
- ImageJ-ready 2‑channel overlays
- JSON files recording parameters

Pipeline:
1. Pre-align stacks (Napari)
2. Annotate damaged sections + select `best_z`
3. Build montage reference volume
4. Register volumes with BigStream (global + distributed piecewise; local or SLURM)

---

## Repository contents

#### `pre-processing/napari_pre-alignment.py`
Interactive Napari tool to rotate/flip stacks for consistent orientation.

#### `pre-processing/annotate_damaged_sections.py`
Napari-based QC tool to flag damaged stacks and select `best_z`. Generates `section_annotations.tsv` and a PDF summary.

#### `pre-processing/montage_register_prealigned.py`
Builds a reference montage from the longest contiguous run of non-damaged sections, trimming around `best_z` and concatenating along Z.

#### `BigStream_register_*.py`
Global + distributed piecewise registration drivers using BigStream (notebook on a local workstation or SLURM cluster). Writes:

---

## Typical usage

### 0) Convert raw data to TIFF

Convert raw files (e.g. `.oir`) to TIFF using Fiji → Process → Batch → Convert.

---

### 1) Pre-align stacks (Napari)

```bash
python pre-processing/napari_pre-alignment.py
```

### 2) QC and damaged-section detection

```bash
# interactive annotation (Napari)
python3 pre-processing/annotate_damaged_sections.py annotate

# generate/update the PDF report from existing annotations
python3 pre-processing/annotate_damaged_sections.py report

# do both (annotate, then report)
python3 pre-processing/annotate_damaged_sections.py all
```

### 3) Montage clean sections

```bash
python3 pre-processing/montage_register_prealigned.py
```

Builds a clean reference volume from non-damaged stacks.

### 4) Register volumes with BigStream (local or SLURM)

```
---

## Environment

```bash
conda activate bigstream
pip install bigstream
```

---