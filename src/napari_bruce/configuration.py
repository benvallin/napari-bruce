# %% Import required libraries ----

import os
import sys
import subprocess
import importlib.resources
import json
import shutil
from pathlib import Path

# %% get_config_file_path() ----

def get_config_file_path() -> str:
  
  """Get path to config file.
  
  Returns:
    str: path to config file expected location on disk.

  """
  
  output = os.path.join(importlib.resources.files('napari_bruce'), 'config.json')
  
  return output

# %% list_stardist_models() ----

def list_stardist_models() -> dict:
  
  """List available StarDist models.
  
  Returns:
    dict: dict with keys and values representing model name and source, respectively.
  
  """
  
  pretrained = ['2D_versatile_fluo', '2D_paper_dsb2018', '2D_demo']
  
  pretrained = {x:'pretrained' for x in pretrained}
  
  user_defined = Path(os.path.join(importlib.resources.files('napari_bruce'),  
                                   'stardist_models'))
    
  if not (os.path.exists(user_defined) and os.path.isdir(user_defined)):
    
    user_defined = {}
  
  else:
    
    user_defined = [x.stem for x in user_defined.iterdir() if x.is_dir()]
    
    user_defined = {x:'user-defined' for x in user_defined}
  
  output = {**pretrained, **user_defined}
  
  return output

# %% check_config_integrity() ----

class ConfigError(RuntimeError):
    
    """Raised when the napari-bruce configuration is invalid."""
    
    pass
  
def check_config_integrity(config: dict) -> None:

  """Check the integrity of the config dict.
  
  Args:
    config: dict representing configuration.
  
  Side effects:
    Raises a ConfigError if config is not a dict or a malformed dict.

  """
      
  if not isinstance(config, dict):
    
    raise ConfigError(f'JSON stored in config file is not a dict.')
  
  def check_dict_kv(exp_kv, in_dict, dict_nm):
    
    missing_keys = [x for x in exp_kv.keys() if x not in in_dict.keys()]
    
    if len(missing_keys) > 0:
      
      tmp = ', '.join(missing_keys)
      
      raise ConfigError(f'Missing {dict_nm} key(s): {tmp}.')
    
    invalid_vals = [x for x in exp_kv.keys() if not isinstance(in_dict[x], exp_kv[x])]
    
    if len(invalid_vals) > 0:
      
      tmp = ', '.join(invalid_vals)
      
      raise ConfigError(f'Invalid value type for {dict_nm} key(s): {tmp}.')
    
    return
  
  exp_main_kv = {'in_dir_path': str, 
                 'out_dir_path': str, 
                 'channels': dict, 
                 'min_pct_ovl_ch0_by_ch1': (int, float),
                 'min_pct_ovl_ch1_by_ch0': (int, float)}
  
  exp_ch_kv = {'0': dict,
               '1': dict}
  
  exp_subch_kv = {'name': str, 
                  'low_pct': (int, float), 
                  'high_pct': (int, float), 
                  'stardist_model': str, 
                  'min_n_pix': (int, float), 
                  'color': str}
  
  check_dict_kv(exp_kv=exp_main_kv, 
                in_dict=config, 
                dict_nm='config main')
  
  for i in ['min_pct_ovl_ch0_by_ch1', 'min_pct_ovl_ch1_by_ch0']:
    
    if not (config[i] >= 0 and config[i] <= 100):
       
       raise ConfigError(f"config['{i}'] must be between 0 and 100.")
  
  check_dict_kv(exp_kv=exp_ch_kv, 
                in_dict=config['channels'], 
                dict_nm='config channels')
  
  available_models = [k for k in list_stardist_models().keys()]
  
  for i in config['channels'].keys():
    
    check_dict_kv(exp_kv=exp_subch_kv, 
                  in_dict=config['channels'][i], 
                  dict_nm=f'config channel {i}')
    
    if config['channels'][i]['stardist_model'] not in available_models:
      
      raise ConfigError(f"config['channels']['{i}']['stardist_model'] must be one of {', '.join(available_models)}.")
    
    for j in ['low_pct', 'high_pct']:
      
      if not (config['channels'][i][j] >= 0 and config['channels'][i][j] <= 100):
        
        raise ConfigError(f"config['channels']['{i}']['{j}'] must be between 0 and 100.")
  
  return

# %% make_default_config() ----

def make_default_config() -> dict:

  """Generate default configuration and write to config.json file.
  
  Returns:
    dict: dict representing default configuration.
  
  Side effects:
    Writes default configuration to config.json file. This function overwrites the file if it already exists.

  """
      
  path = get_config_file_path()
  
  in_dir_path = str(Path.home())
  
  out_dir_path = str(os.path.join(in_dir_path, 'napari_bruce_results'))
      
  output = {'in_dir_path': in_dir_path,
            'out_dir_path': out_dir_path,
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
# %% get_config() ----

def get_config() -> dict:

  """Read configuration from config.json file and check integrity.
  
  Returns:
    dict: dict representing configuration. If config.json file does not exist, creates it and returns default config.

  """
    
  path = get_config_file_path()
  
  if os.path.exists(path):
    
    try:
      
      with open(path, 'r', encoding='utf-8') as f:
        
        output = json.load(f)
        
    except json.JSONDecodeError:
      
      raise ValueError(f'Invalid JSON in config file.')

    except OSError:
      
      raise OSError(f'Could not read config file.')
    
    check_config_integrity(config=output)
        
  else:
    
    output = make_default_config()
  
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

# %% add_stardist_model() ----

def add_stardist_model(src: str) -> None:
  
  """Add StarDist model to napari-bruce.
  
  Args:
    src: path to input StarDist model directory.
  
  Side effects:
    Copies the input StarDist model directory into the napari-bruce src directory.

  """
  
  in_dir_path = Path(src).expanduser()
  
  if not os.path.exists(in_dir_path):
    
    raise FileNotFoundError(f'Input path does not exist: {in_dir_path}.')
  
  if not os.path.isdir(in_dir_path):
    
    raise NotADirectoryError(f'Input path is not a directory: {in_dir_path}.')
  
  out_dir_path = os.path.join(os.path.join(importlib.resources.files('napari_bruce'), 
                                           'stardist_models'),
                              Path(in_dir_path).stem)
  
  shutil.copytree(src=in_dir_path, 
                  dst=out_dir_path, 
                  copy_function=shutil.copy2, 
                  dirs_exist_ok=True)
