# ***Bruce***

A napari plugin for PALM RoboSoftware 4.5 image analysis using StarDist segmentation.

---

## Features

- Load PALM `.zvi` images
- StarDist-based cell segmentation
- Manual ROI editing
- Overlap analysis between channels
- Export PALM-compatible ROI files

---

## Installation

### Windows

Install miniforge from: https://github.com/conda-forge/miniforge<br>

```bash
# Create environment
mamba env create -f napari-bruce_windows_native.yml
# Activate environment
mamba activate napari-bruce
# Install napari-bruce
python -m pip install "git+https://github.com/benvallin/napari-bruce.git". 
# Run napari-bruce 
bruce -h
```