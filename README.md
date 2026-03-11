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
Global + distributed piecewise registration drivers using **BigStream** (local workstation or SLURM cluster). Writes:
- warped volumes
- ImageJ-ready 2‑channel overlays
- timestamped JSON configs (paths + parameters)

---

## References

- BigStream (Janelia SciComp): https://github.com/JaneliaSciComp/bigstream
- Marquez-Legorreta *et al.* (2026). *Whole-Brain Co-Mapping of Gene Expression and Neuronal Activity at Cellular Resolution in Behaving Zebrafish*. bioRxiv 2026.02.07.704095. https://doi.org/10.64898/2026.02.07.704095

---

## Environment

```bash
conda activate bigstream
pip install bigstream
```

---