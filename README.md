# ***Bruce***

A napari plugin for drawing PALM RoboSoftware elements using StarDist segmentation.

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

Platform-specific `.yml` files to create the bruce-env environment containing all required dependencies are available in `env`.<br>
- Windows (native): `napari-bruce_windows_native.yml`<br>
- macOS (ARM): `napari-bruce_macos_arm.yml`<br>
- linux: `napari-bruce_linux.yml`<br>

### Windows

In a Miniforge Prompt:<br>
```bash
# Create environment
mamba env create -f napari-bruce_windows_native.yml
# Activate environment
mamba activate bruce-env
# Install napari-bruce
python -m pip install "git+https://github.com/benvallin/napari-bruce.git". 
# Run napari-bruce 
bruce -h
```