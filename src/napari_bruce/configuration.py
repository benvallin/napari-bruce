# %% Import required libraries ----

import os
import sys
import subprocess
import importlib.resources
import json
from pathlib import Path

# %% get_config_file_path() ----

def get_config_file_path() -> str:
  
  """Get path to config file.
  
  Returns:
    str: path to config file location on disk.

  """
  
  output = os.path.join(importlib.resources.files('napari_bruce'), 'config.json')
  
  return output

# %% get_config() ----

def get_config() -> dict:

  """Read configuration from config.json file.
  
  Returns:
    dict: dict representing configuration.

  """
    
  path = get_config_file_path()
  
  if os.path.exists(path):
    
    with open(path, 'r') as f:
      
      output = json.load(f)
    
    return output
  
  else:
  
    return None

# %% make_default_config() ----

def make_default_config() -> dict:

  """Generate default configuration and write to config.json file.
  
  Returns:
    dict: dict representing default configuration.
  
  Side effects:
    Writes default configuration to config.json file. This function overwrites the file if it already exists.

  """
      
  path = get_config_file_path()
    
  output = {'in_dir_path': '~/Documents/rwm_lab/lcmseq_project/2025.10.31_palm_image/',
            'out_dir_path': '~/Desktop/napari_bruce_output/',
            'stardist_models_dir_path': '~/Documents/rwm_lab/lcmseq_project/model_training/',
            'channels': {0: {'name': 'TH', 
                               'low_pct': 0.05,
                               'high_pct': 99.95,
                               'stardist_model': 'stardist_th',
                               'min_n_pix': 800,
                               'color': 'purple'},
                         1: {'name': 'pSyn', 
                               'low_pct': 0.05,
                               'high_pct': 99.95,
                               'stardist_model': 'stardist_psyn',
                               'min_n_pix': 800,
                               'color': 'cyan'}},
            'min_pct_ovl_ch0_by_ch1': 20,
            'min_pct_ovl_ch1_by_ch0': 80}
  
  with open(path, 'w') as f:
    
    json.dump(output, f, indent=2)
  
  return output

# %% open_in_editor() ----

def open_in_editor(path: str) -> None:
  
  """Open 'path' in the system's default editor / viewer.
  
  Args:
    path (str): path to file.
    
  """
  
  path = str(Path(path).resolve())
  
  if sys.platform == 'darwin':  # macOS
        
    subprocess.run(['open', '-a', 'TextEdit', path], check=False)
  
  elif sys.platform.startswith('win'):  # Windows
    
    subprocess.run(['notepad.exe', path], check=False)
  
  else:  # Linux / others
    
    subprocess.run(['xdg-open', path], check=False)
    