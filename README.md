# ***🦇  Bruce***

A **napari plugin for drawing PALM RoboSoftware elements using StarDist segmentation**

---

## Quick start

After installation (see below):

```bash
# Launch plugin from CLI:
bruce

# List available options with:
bruce -h
```

---

## Features

- Load 2-channel images and metadata from `.zvi` files produced by PALM RoboSoftware 4.5
- Perform StarDist-based cell segmentation (default or user-defined models)
- Allow manual editing of ROIs / elements in napari
- Perform ROI overlap analysis between 2 channels
- Export element list as `.txt` file compatible with PALM RoboSoftware 4.5

---

## System requirements

- **Conda / Mamba** (recommended)
  - Install Miniforge from: https://github.com/conda-forge/miniforge
- **Java (OpenJDK)** – required for Bio-Formats `.zvi → OME-TIFF` conversion
- **GPU (optional)** – for accelerated StarDist inference

---

## Installation

Bruce requires a **platform-specific Conda environment** due to differences in GPU support and native dependencies (TensorFlow, CUDA, Java).<br>
Predefined environment files are provided in the `env/` directory:
```md
| Platform | Environment file |
|--------|------------------|
| Windows (native) | `env/bruce-env_windows_native.yml` |
| macOS (Apple Silicon) | `env/bruce-env_macos_arm.yml` |
| Linux | `env/bruce-env_linux.yml` |
```

```md
Open a terminal and run:

```bash
# Create the conda environment (replace <ENV_FILE> with the appropriate .yml file)
mamba env create -f <ENV_FILE>

# Activate the environment
mamba activate bruce-env

# Install Bruce from GitHub
python -m pip install "git+https://github.com/benvallin/napari-bruce.git"

# Launch Bruce via the command line
bruce

# Or launch Bruce directly from napari
napari --with napari-bruce

---

## Configuration

Bruce stores its configuration in a user-specific JSON file.

Useful commands:

```bash
# Show config file path
bruce --show-config-path

# Open config in default editor
bruce --edit-config

# Reset config to defaults
bruce --reset-config

# List available StarDist models
bruce --list-models

```