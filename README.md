# ***🦇 Bruce***

**A napari plugin for creating PALMRobo elements using StarDist segmentation**


---

## Description

**Bruce** is a napari plugin for automated detection of cells of interest in PALM RoboSoftware images, enabling fast laser capture microdissection on the PALM MicroBeam system.

Key features:
- Load 2-channel images and metadata from `.zvi` files produced by PALMRobo 4.9
- Perform StarDist-based cell segmentation (default or user-defined models)
- Allow manual editing of ROIs / elements in napari
- Perform ROI overlap analysis between 2 channels
- Export element list as `.txt` file compatible with PALMRobo 4.9


---


## System requirements

- **Conda / Mamba** (recommended)
  - Install Miniforge from: https://github.com/conda-forge/miniforge
- **Java (OpenJDK)** – required for Bio-Formats `.zvi → OME-TIFF` conversion
- **GPU (optional)** – for accelerated StarDist inference


---


## Installation

Bruce requires a **platform-specific Conda environment** due to differences in native dependencies and GPU support. Predefined environment files are provided in the `env/` directory:

```md
| Platform                | Environment file                     |
|-------------------------|--------------------------------------|
| Windows (native)        | `env/bruce-env_windows_native.yml`   |
| macOS (Apple Silicon)   | `env/bruce-env_macos_arm.yml`        |
```

Open a terminal and run:

```bash
# Create the conda environment (replace <ENV_FILE> with the appropriate .yml file)
mamba env create -f <ENV_FILE>

# Activate the environment
mamba activate bruce-env

# Install Bruce from PyPI
pip install napari-bruce

# Launch Bruce via the command line
bruce

# Or launch Bruce directly from napari
napari --with napari-bruce

# Show help
bruce -h
```


---


## Configuration

Bruce stores its configuration in a user-specific JSON file.

Available commands:

```bash
# Show config file path
bruce --show-config-path

# Open config in default editor
bruce --edit-config

# Reset config to defaults
bruce --reset-config
```


---


## GPU support & StarDist models 

Bruce runs StarDist predictions on the GPU when visible to TensorFlow, and supports user-defined StarDist models.

Available commands:

```bash
# Check whether GPU(s) are visible to TensorFlow
bruce --gpu-status

# List available StarDist models
bruce --list-models

# Add a user-defined StarDist model located at <MODEL_DIR>
bruce --add-model <MODEL_DIR>
```


---


## Image annotation for model training 

Bruce comes with a dedicated GUI to annotate 2-channel images produced by PALMRobo 4.9.

Available commands:

```bash
# Construct an image set from 2-channel .zvi files located at <IN_DIR> and save to <OUT_DIR>/imgs.pkl
bruce --zvi-to-dict <IN_DIR> <OUT_DIR>

# Open the annotation GUI to label the image set at <IN_DIR>/imgs.pkl (or <IN_DIR>/imgs_annotated.pkl if exists) and save image/mask pairs to <OUT_DIR>/imgs_annotated.pkl
bruce --annotate <IN_DIR> <OUT_DIR>
```


---


## Example images

Example `.zvi` files are provided in the `example_images/` directory for testing and demonstration purposes.
They are not installed with the napari-bruce Python package.
