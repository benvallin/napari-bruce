# Download miniforge
https://github.com/conda-forge/miniforge. 

# Create environment
mamba env create -f napari-bruce_windows_native.yml. 
mamba activate napari-bruce. 
potentially install git. 
# Install napari-bruce 
python -m pip install "git+https://github.com/benvallin/napari-bruce.git". 

# Run napari-bruce 
bruce -h. 