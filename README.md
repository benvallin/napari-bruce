# ***Bruce***

A napari plugin for the analysis of PALM RoboSoftware 4.5 images using StarDist segmentation.

---

## Features

- Load `.zvi` images from PALM RoboSoftware 4.5
- StarDist-based cell segmentation
- Manual ROI editing
- Overlap analysis between channels
- Export PALM-compatible ROI files

---

## System requirements

- Mamba (preferred) / Conda<br>
The Miniforge can be installed from: https://github.com/conda-forge/miniforge

---

## Installation

### Windows

In a Miniforge Prompt:<br>
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