# ***🦇 Bruce***

A **napari plugin** for drawing **PALM RoboSoftware** elements using **StarDist** segmentation

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
Predefined `.yml` environment files are provided in the `env/` directory:
```md
| Platform | Environment file |
|--------|------------------|
| Windows (native) | `env/bruce-env_windows_native.yml` |
| macOS (Apple Silicon) | `env/bruce-env_macos_arm.yml` |
| Linux | `env/bruce-env_linux.yml` |
```

Open a terminal and run:
```bash
# Create environment (replace `<ENV_FILE>` with the appropriate `.yml` file)
mamba env create -f <ENV_FILE>

# Activate environment
mamba activate bruce-env

# Install Bruce
python -m pip install "git+https://github.com/benvallin/napari-bruce.git"

# Launch plugin from CLI:
bruce

# Alternatively, launch plugin from napari:
napari --with napari-bruce
```