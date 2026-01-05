# %% Import required libraries ----

import os
import math
import numpy as np
import pandas as pd
import subprocess
import shutil
import tifffile
import cv2
import pickle
import ome_types
import importlib.resources
from itertools import groupby
from typing import Any
from pathlib import Path
from datetime import datetime

# %% robust_normalization() ----

def robust_normalization(img: np.ndarray, 
                         low_pct: float = 0.05, 
                         high_pct: float = 99.95) -> np.ndarray:
  
  """Perform robust normalization (contrast stretching with percentile clipping).
  
  Args:
    img (numpy.ndarray): image to normalize.
    low_pct (float): lower percentile to clip.
    high_pct (float): upper percentile to clip.
  
  Returns:
    numpy.ndarray: normalized image.

  """
  
  # Clip to low / high percentiles 
  low, high = np.percentile(img, (low_pct, high_pct))
  
  # Normalize to 0–255 (uint8)
  output = cv2.normalize(np.clip(img, low, high), 
                         None, 
                         0, 
                         255, 
                         cv2.NORM_MINMAX).astype(np.uint8)
  
  return output

# %% msk_to_cnts() ----

def msk_to_cnts(msk: np.ndarray) -> np.ndarray:
  
  """Get contours of submasks in mask.
  
  Args:
    msk (numpy.ndarray): mask (0=no cells, 1=first cell, 2=second cell...).
  
  Returns:
    list: list of submasks contours as pixel coordinates.

  """
  
  output = []
  
  for i in np.unique(msk)[1:]:
    
    m = msk == i
    
    if m.sum() > 0:
      
      c = cv2.findContours(m.astype(np.uint8), 
                           mode=cv2.RETR_EXTERNAL, 
                           method=cv2.CHAIN_APPROX_NONE)
      
      c = c[0][0].squeeze()
      
      if len(c) > 2:
        
        output.append(c)
      
  return output

# %% count_submsks_pixels() ----

def count_submsks_pixels(msk: np.ndarray) -> dict:
  
  """Count submasks pixels in mask.
  
  Args:
    msk (numpy.ndarray): mask (0=no cells, 1=first cell, 2=second cell...).
  
  Returns:
    dict: dict of pixel counts for each submask ID.

  """
  
  flat = msk.ravel()
  
  counts = np.bincount(flat)
  
  output = {int(i): int(counts[i]) for i in np.nonzero(counts)[0] if i != 0}
  
  return output

# %% discard_small_submsks() ----

def discard_small_submsks(msk: np.ndarray, 
                          pix_dict: dict = None, 
                          min_n_pix: int = 100) -> np.ndarray:
  
  """Discard small submasks from mask.
  
  Args:
    msk (numpy.ndarray): mask (0=no cells, 1=first cell, 2=second cell...).
    pix_dict (dict): dict of pixel counts for each submask ID.
    min_n_pix (int): minimum number of pixels for a submask to be kept.
  
  Returns:
    numpy.ndarray: mask array with small submasks removed.

  """
  
  if pix_dict is None:
    
    pix_dict = count_submsks_pixels(msk=msk)
  
  discard = [k for k, v in pix_dict.items() if v < min_n_pix]
  
  output = np.where(np.isin(msk, discard), 0, msk)
  
  return output

# %% append_shapes_to_msk() ----

def append_shapes_to_msk(msk: np.ndarray,
                         shapes: list,
                         start_idx: int) -> np.ndarray:
  
  """Append shapes to mask.
  
  Args:
    msk (numpy.ndarray): mask (0=no cells, 1=first cell, 2=second cell...).
    shapes (list): list of ndarrays of shape (y, x) representing shapes to close and draw.
    start_idx (int): starting index for the first shape to draw.
  
  Returns:
    numpy.ndarray: mask supplemented with closed shapes.

  """
  
  idx = start_idx
  
  output = msk.copy()
  
  for i in shapes.copy():
    roi = np.vstack([i, i[0]])
    roi = roi[:, ::-1].astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(output, [roi], idx)
    idx += 1
    
  return output

# %% make_submsks() ----

def make_submsks(msk: np.ndarray) -> dict:
  
  """Make submasks from main mask.
  
  Args:
    msk (numpy.ndarray): mask (0=no cells, 1=first cell, 2=second cell...).
  
  Returns:
    dict: dict of submasks, with keys and values representing IDs and arrays of True/False, respectively.

  """
  
  output = {}
  
  for i in np.unique(msk)[1:]:
    
    output[i] = msk == i
  
  return output

# %% get_submsk_ids_ovl_by_mainmsk() ----

def get_submsk_ids_ovl_by_mainmsk(submsks: dict, 
                                  msk: np.ndarray) -> list:
  
  """Get the ID of submasks overlapping with a main mask.
  
  Args:
    submsks (dict): dict of submasks with keys and values representing IDs and arrays of True/False, respectively.
    msk (numpy.ndarray): main mask (0=no cells, 1=first cell, 2=second cell...).
  
  Returns:
    list: list of IDs corresponding to the submasks overlapping with the main mask.

  """
  
  submsks_mainmsk_overlap = {}
  
  for i in submsks.keys():
    submsks_mainmsk_overlap[i] = np.any((submsks[i] == True) & (msk > 0))
  
  output = [k for k, v in submsks_mainmsk_overlap.items() if v == True]
  
  return output

# %% get_pct_submsks1_ovl_by_submsks2() ----

def get_pct_submsks1_ovl_by_submsks2(submsks1: dict, 
                                     submsks2: dict, 
                                     submsks1_pix_dict: None | dict = None) -> dict:
  
  """Get the percentage of submask1 area overlapped by submask2 for every submask1-submask2 pair.
  
  Args:
    submsks1 (dict): dict of submasks1 with keys and values representing IDs and ndarrays of True/False, respectively.
    submsks2 (dict): dict of submasks2 with keys and values representing IDs and ndarrays of True/False, respectively.
    submsks1_pix_dict (dict): dict of pixel counts for each submask1 ID.
  
  Returns:
    dict: percentage of submask1 area overlapped by submask2 for every submask1-submask2 pair. 
    Main keys represent submask1 IDs, inner keys and values represent submask2 IDs and percentages of overlapped submask1 area, respectively.

  """
  
  if submsks1_pix_dict is None:
    
    get_submsk1_area = lambda k: np.sum(submsks1[k] == True)
      
  else:
    
    get_submsk1_area = lambda k: submsks1_pix_dict[k]
      
  output = {}
  
  for k in submsks1.keys():
    
    output[k] = {}
    
    submsks1_sum = get_submsk1_area(k)
    
    for k1 in submsks2.keys():
      
      submsks1_submsks2_sum = np.sum((submsks1[k] == True) & (submsks2[k1] == True))
      
      output[k][k1] = submsks1_submsks2_sum / submsks1_sum * 100
    
  return output

# %% unify_pct_ovl_dicts() ----

def unify_pct_ovl_dicts(pct_ovl_dict1: dict, 
                        pct_ovl_dict2: dict,
                        min_pct_ovl_1by2: int | float,
                        min_pct_ovl_2by1: int | float) -> dict:
  
  """Unify percentage overlap dicts.
  
  Args:
    pct_ovl_dict1 (dict): dict representing the percentage of submask1 area overlapped by submask2 for every submask1-submask2 pair.
    pct_ovl_dict2 (dict): dict representing the percentage of submask2 area overlapped by submask1 for every submask2-submask1 pair.
    min_pct_ovl_1by2 (int | float): minimal percentage of submask1 area to be overlapped with a submask2 to record submask1 as submask2-positive.
    min_pct_ovl_2by1 (int | float): minimal percentage of submask2 area to be overlapped with a submask1 to record submask1 as submask2-positive.
  
  Returns:
    dict: unified dict with main keys representing submask1 IDs and values containing 'status', 'summary' and 'data'. 

  """
  
  output = {}
  
  for i in pct_ovl_dict1.keys():
    
    for j in pct_ovl_dict2.keys():
      
      pct_ovl_1by2 = pct_ovl_dict1[i][j]
      pct_ovl_2by1 = pct_ovl_dict2[j][i]
      mean_pct_ovl = np.mean([pct_ovl_1by2, pct_ovl_2by1])
      pct_ovl_1by2_pass = pct_ovl_1by2 >= min_pct_ovl_1by2
      pct_ovl_2by1_pass = pct_ovl_2by1 >= min_pct_ovl_2by1
      pct_ovl_pass = pct_ovl_1by2_pass & pct_ovl_2by1_pass
      
      tmp = {j: {'pct_ovl_1by2': pct_ovl_1by2,
                 'pct_ovl_2by1': pct_ovl_2by1,
                 'mean_pct_ovl': mean_pct_ovl,
                 'pct_ovl_pass': pct_ovl_pass}}
      
      if i not in output.keys():
        
        output[i] = {'data': tmp}
        
      else:
        
        output[i]['data'].update(tmp)
  
  for i in output.keys():
    
    pct_ovl_pass = []
    pct_ovl_1by2 = []
    pct_ovl_2by1 = []
    mean_pct_ovl = []
    
    for j in output[i]['data'].keys():
      
      pct_ovl_pass.append(output[i]['data'][j]['pct_ovl_pass'])
      pct_ovl_1by2.append(output[i]['data'][j]['pct_ovl_1by2'])     
      pct_ovl_2by1.append(output[i]['data'][j]['pct_ovl_2by1']) 
      mean_pct_ovl.append(output[i]['data'][j]['mean_pct_ovl']) 
  
    n_ovl_pass = sum(pct_ovl_pass)
    status = 'pos' if n_ovl_pass > 0 else 'amb'
    max_pct_ovl_1by2 = np.max(pct_ovl_1by2)
    max_pct_ovl_2by1 = np.max(pct_ovl_2by1)
    max_mean_pct_ovl = np.max(mean_pct_ovl)
    
    output[i]['summary'] = {'n_ovl_pass': n_ovl_pass,
                            'max_pct_ovl_1by2': max_pct_ovl_1by2,
                            'max_pct_ovl_2by1': max_pct_ovl_2by1,
                            'max_mean_pct_ovl': max_mean_pct_ovl}
  
    output[i]['status'] = status
  
    output[i] = {k:output[i][k] for k in ['status', 'summary', 'data']}
  
  return output

# %% get_submsks1_submsks2_status() ----

def get_submsks1_submsks2_status(msk1: np.ndarray, 
                                 msk2: np.ndarray, 
                                 submsks1_pix_dict: dict,
                                 submsks2_pix_dict: dict,
                                 min_pct_ovl_1by2: int | float,
                                 min_pct_ovl_2by1: int | float) -> dict:
  
  """Get submasks1 / submasks2 status (negative, positive or ambiguous for each others).
  
  Args:
    msk1 (numpy.ndarray): primary mask (0=no cells, 1=first cell, 2=second cell...).
    msk2 (numpy.ndarray): secondary mask (0=no cells, 1=first cell, 2=second cell...).
    submsks1_pix_dict (dict): dict of pixel counts for each submask1 ID.
    submsks2_pix_dict (dict): dict of pixel counts for each submask2 ID.
    min_pct_ovl_1by2 (int | float): minimal percentage of submask1 area to be overlapped with a submask2 to record submask1 and submask2 as submask2-positive and submask1-positive.
    min_pct_ovl_2by1 (int | float): minimal percentage of submask2 area to be overlapped with a submask1 to record submask1 and submask2 as submask2-positive and submask1-positive.
  
  Returns:
    dict: dict of list of primary submasks IDs, with keys representing double positive status and values representing primary submasks IDs.

  """
  
  submsks1 = make_submsks(msk=msk1)
    
  ids1_ovl_by2 = get_submsk_ids_ovl_by_mainmsk(submsks=submsks1, msk=msk2)
  
  ids1_no_ovl = [k for k in submsks1.keys() if k not in ids1_ovl_by2]
  
  submsks1_ovl_by2 = {k:v for k, v in submsks1.items() if k in ids1_ovl_by2}
  
  submsks2 = make_submsks(msk=msk2)
  
  ids2_ovl_by1 = get_submsk_ids_ovl_by_mainmsk(submsks=submsks2, msk=msk1)
  
  ids2_no_ovl = [k for k in submsks2.keys() if k not in ids2_ovl_by1]
  
  submsks2_ovl_by1 = {k:v for k, v in submsks2.items() if k in ids2_ovl_by1}
    
  pct_ovl_1by2 = get_pct_submsks1_ovl_by_submsks2(submsks1=submsks1_ovl_by2, 
                                                  submsks2=submsks2_ovl_by1,
                                                  submsks1_pix_dict=submsks1_pix_dict)
  
  pct_ovl_2by1 = get_pct_submsks1_ovl_by_submsks2(submsks1=submsks2_ovl_by1, 
                                                  submsks2=submsks1_ovl_by2,
                                                  submsks1_pix_dict=submsks2_pix_dict)

  prim_sec_neg = {i:{'status': 'neg'} for i in ids1_no_ovl}
  
  sec_prim_neg = {i:{'status': 'neg'} for i in ids2_no_ovl} 
  
  prim_sec_ovl = unify_pct_ovl_dicts(pct_ovl_dict1=pct_ovl_1by2, 
                                     pct_ovl_dict2=pct_ovl_2by1,
                                     min_pct_ovl_1by2=min_pct_ovl_1by2,
                                     min_pct_ovl_2by1=min_pct_ovl_2by1)
  
  prim_sec_pos = {k:v for k, v in prim_sec_ovl.items() if v['status'] == 'pos'}
  prim_sec_amb = {k:v for k, v in prim_sec_ovl.items() if v['status'] == 'amb'}
  
  sec_prim_ovl = unify_pct_ovl_dicts(pct_ovl_dict1=pct_ovl_2by1, 
                                     pct_ovl_dict2=pct_ovl_1by2,
                                     min_pct_ovl_1by2=min_pct_ovl_2by1,
                                     min_pct_ovl_2by1=min_pct_ovl_1by2)

  sec_prim_pos = {k:v for k, v in sec_prim_ovl.items() if v['status'] == 'pos'}
  sec_prim_amb = {k:v for k, v in sec_prim_ovl.items() if v['status'] == 'amb'}
  
  dicts = [prim_sec_pos, prim_sec_amb, sec_prim_pos, sec_prim_amb]
  
  sorted_dicts = [
    dict(sorted(d.items(), 
                key=lambda x: (x[1]['summary']['max_mean_pct_ovl'],
                               x[1]['summary']['max_pct_ovl_1by2'],
                               x[1]['summary']['max_pct_ovl_2by1']), 
                reverse=True))
    for d in dicts
  ]
  
  prim_sec_pos, prim_sec_amb, sec_prim_pos, sec_prim_amb = sorted_dicts

  prims = {'neg': prim_sec_neg,
           'pos': prim_sec_pos,
           'amb': prim_sec_amb}
  
  secs = {'neg': sec_prim_neg,
          'pos': sec_prim_pos,
          'amb': sec_prim_amb}
  
  def add_area(status_dict, submsks_pix_dict):
    
    for i in status_dict.keys():
    
      for j in status_dict[i].keys():
      
        status_dict[i][j]['area'] = submsks_pix_dict[j]
  
    for i in ['pos', 'amb']:
    
      for j in status_dict[i].keys():
      
        status_dict[i][j] = {k:status_dict[i][j][k] for k in ['status', 'area', 'summary', 'data']}
  
  
    status_dict['neg'] = dict(sorted(status_dict['neg'].items(), 
                                     key=lambda x: (x[1]['area']), 
                                     reverse=True))
  
    return 
  
  add_area(status_dict=prims, 
           submsks_pix_dict=submsks1_pix_dict)
  
  add_area(status_dict=secs, 
           submsks_pix_dict=submsks2_pix_dict)
  
  output = (prims, secs)
  
  return output

# %% status_dict_to_submsks_and_cnts() ----

def status_dict_to_submsks_and_cnts(msk: np.ndarray, 
                                    status_dict: dict) -> dict:
  
  """Get submasks and contours from main mask and submasks status dict.
  
  Args:
    msk (numpy.ndarray): mask (0=no cells, 1=first cell, 2=second cell...).
    status_dict (dict): submasks status dict produced by get_submsks1_submsks2_status().
  
  Returns:
    dict: dict of submasks and contours for every submask ID in submasks status dict. 
    Main keys represent submask status (negative, positive or ambiguous).

  """
  
  tmp = {}
  
  for i in status_dict.keys():
    
    ids = list(status_dict[i].keys())
    
    submsk = np.where(np.isin(msk, ids),
                      msk,
                      0)
    
    cnt = msk_to_cnts(msk=submsk)
    
    tmp[i] = {'msk': submsk, 'cnt': cnt}
    
  submsks = {k:v['msk'] for k, v in tmp.items()}
  cnts = {k:v['cnt'] for k, v in tmp.items()}
  
  output = (submsks, cnts)
  
  return output

# %% make_picklable() ----

def make_picklable(data: Any) -> Any:
  
  """Make an object pickle-able.
  
  Args:
    data: object to make pickle-able.
  
  Returns:
    A deep-copy of data containing only picklable primitives and numpy arrays. If a napari layer object is found, tries to extract .data if present.

  """

  def sanitize(obj): 
    # Basic safe types
    if obj is None or isinstance(obj, (int, float, str, bool)):
      return obj
    if isinstance(obj, np.ndarray):
      return obj  # Arrays are picklable
    if isinstance(obj, (list, tuple)):
      return type(obj)(sanitize(x) for x in obj)
    if isinstance(obj, dict):
      return {k: sanitize(v) for k, v in obj.items()}
    # Napari layers commonly have .data attribute we can use
    if hasattr(obj, 'data'):
      try:
        d = getattr(obj, 'data')
        # If it's a napari layer, d is usually numpy array
        if isinstance(d, np.ndarray):
          return d.copy()
      except Exception:
        pass
    # Fallback: try pickle; if that fails, replace with a string summary
    try:
      pickle.dumps(obj)
      return obj
    except Exception:
      return f"<Unserializable:{type(obj).__name__}>"
  
  output = sanitize(obj=data)    
  
  return output

# %% pickle_data() ----
    
def pickle_data(data: Any, 
                filename: str) -> None:
  
  """Make an object pickle-able and write it to pickle file.
  
  Args:
    data: object to make pickle-able and write to file.
    filename (str): file name.
  
  Side effects:
    Write object to pickle file.

  """
  
  sanitized = make_picklable(data=data) 
  
  with open(filename, 'wb') as f:
    
    pickle.dump(sanitized, f, protocol=pickle.HIGHEST_PROTOCOL)

# %% require_java() ----

def require_java() -> str:
  
  """Look for java executable.
    
  Returns:
    str: java executable. 

  """
  java_exe = shutil.which('java')
  
  if java_exe is None:
    
    raise RuntimeError('java not found on PATH; please install OpenJDK.')
  
  return java_exe
  
# %% convert_zvi_to_ome() ----
    
def convert_zvi_to_ome(file: str, 
                       out_dir_path: str, 
                       jar_pkg: str = 'napari_bruce.bioformats',
                       jar_name: str = 'bioformats_package.jar', 
                       java_opts: list | None = None,
                       timeout: float | None = 300.0) -> None:
  
  """Convert PALM .zvi file to OME-TIFF using Bio-Formats ImageConverter in headless mode.
  
  Args:
    file (str): path to PALM .zvi file.
    out_dir_path (str): path to output directory. The directory is created if it does not already exist.
    jar_pkg (str): name of the Python package containing the Bio-Formats JAR file.
    jar_name (str): file name of the Bio-Formats JAR inside 'jar_pkg'. 
    java_opts (list): additional JVM options to pass to the java command.
    timeout (float): maximum number of seconds to allow the conversion subprocess to run.
  
  Side effects:
    Write OME-TIFF file to 'out_dir_path' using the base name of the input '.zvi' file and '.ome.tiff' extension. 

  """
  
  # Check Java
  java_exe = require_java()
  
  # Locate jar
  jar_path = importlib.resources.files(jar_pkg) / jar_name
  
  if not jar_path.exists(): 
    
    raise RuntimeError(f'Jar not found: {jar_path}')

  # Build java options
  java_opts = list(java_opts) if java_opts else []
  
  if "-Djava.awt.headless=true" not in java_opts:
    
    java_opts.insert(0, "-Djava.awt.headless=true")
  
  # Define classpath flag
  classpath_flag = ["-cp", str(jar_path)]

  # Define Bio-Formats CLI main class
  main_class = "loci.formats.tools.ImageConverter"

  # Expand input paths  
  in_file_path = str(Path(file).expanduser())
  out_dir_path = str(Path(out_dir_path).expanduser())
    
  # Create output directory if missing
  os.makedirs(out_dir_path, exist_ok=True)
  
  # Construct output file path 
  out_file_path = os.path.join(out_dir_path, Path(file).stem+'.ome.tiff')
  
  # Construct command
  cmd = ([java_exe]
         + java_opts
         + classpath_flag
         + [main_class, '-overwrite', in_file_path, out_file_path])

  # Run conversion
  proc = subprocess.run(cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout)
  
  # Raise error if conversion failed
  if proc.returncode != 0:
    
    raise RuntimeError(f'Image conversion failed (rc={proc.returncode}).\n'
                       f'STDOUT:\n{proc.stdout}\n'
                       f'STDERR:\n{proc.stderr}')

  return

# %% load_ome_tiff() ----

def load_ome_tiff(file: str) -> list:

  """Load OME-TIFF file derived from PALM .zvi file.
  
  Args:
    file (str): path to .ome.tiff file.
  
  Returns:
    list: list containing 'images' and 'metadata' dict objects.

  """
     
  # Process metadata   
  ome = tifffile.TiffFile(file).ome_metadata    
  ome = ome_types.from_xml(str(ome))
  
  metadata = {'img_nm': Path(ome.images[0].name).stem}

  metadata['objectives'] = ome.instruments[0].objectives[0].model_dump()

  pixels = ome.images[0].pixels
  planes = pixels.planes
  channels = pixels.channels

  stage_pos = {}
  for i, j in enumerate(planes):
    stage_pos[i] = {'position_x': j.position_x,
                    'position_y': j.position_y}

  stage_pos_equal = len(list(groupby(stage_pos.values()))) == 1

  if not stage_pos_equal:
    raise ValueError('x / y stage positions are not identical for all channels!')
  
  metadata['image'] = {'size_x': pixels.size_x,
                       'size_y': pixels.size_y,
                       'physical_size_x': pixels.physical_size_x,
                       'physical_size_y': pixels.physical_size_y,
                       'position_x': stage_pos[0]['position_x'],
                       'position_y': stage_pos[0]['position_y']}

  metadata['channels'] = {}
  for i, j in enumerate(zip(channels, planes)):
    
    metadata['channels'][i] = {'name': j[0].name,
                               'excitation_wavelength': j[0].excitation_wavelength,
                               'emission_wavelength': j[0].emission_wavelength,
                               'exposure_time': j[1].exposure_time}
  
  # Process images    
  imgs = tifffile.imread(file)
  images = {}
  
  for k in metadata['channels'].keys():
    nm = metadata['channels'][k]['name']
    images[nm] = {'img': imgs[k, :,:]}
  
  output = (images, metadata)
  
  return output

# %% txt_to_elem_dfs() ----

def txt_to_elem_dfs(file: str) -> list:

  """Load .txt file containing PALM element properties.
  
  Args:
    file (str): path to PALM .txt file.
  
  Returns:
    list: list of element-specific pandas.DataFrame objects.

  """
    
  # Construct pd.DataFrame from elements.txt 
  df = pd.read_csv(filepath_or_buffer=file, 
                   sep='\t',
                   skiprows=7)
  
  # Remove empty row between column names and first element header
  # => This empty row occurs for .txt files produced by napari-bruce but not by PALM RoboSoftware
  if isinstance(df.iloc[0,0], float) and math.isnan(df.iloc[0,0]):
    df = df.iloc[1:, :]
    
  # Record rows corresponding to element headers
  # => Rows where column 'No' does not contains ',' nor a missing value
  df['is_header'] = ~ (df['No'].astype(str).str.contains(',').astype(bool) | 
                       df['No'].isna())
  
  # Compute element IDs
  df['elem_id'] = df['is_header'].cumsum()
  
  # Split dataframe by element ID
  output = [i for _, i in df.groupby('elem_id')]
  
  return output

# %% get_elem_metadata() ----
  
def get_elem_metadata(elem_df: pd.DataFrame) -> dict:

  """Extract element metadata from an element-specific pandas.DataFrame.
  
  Args:
    elem_df (pandas.DataFrame): element-specific pandas.DataFrame.
  
  Returns:
    dict: dict of metadata with keys: 'Type', 'Color', 'Thickness', 'No', 'Laser function', 'CutShot', 'Area', 'Z', 'Well', 'Objective', 'Comment', 'Coordinates'.

  """
    
  output = elem_df.loc[:,
                       ['Type', 'Color', 'Thickness',
                        'No', 'Laser function', 'CutShot', 
                        'Area', 'Z', 'Well', 
                        'Objective', 'Comment', 'Coordinates']] 
    
  output = output.iloc[0, :].to_dict()
  
  return output

# %% get_elem_cnt() ----

def get_elem_cnt(elem_df: pd.DataFrame) -> np.ndarray:

  """Extract element contours from an element-specific pandas.DataFrame.
  
  Args:
    elem_df (pandas.DataFrame): element-specific pandas.DataFrame.
  
  Returns:
    numpy.ndarray: element-specific contours as an array of shape (N, 2).

  """
    
  # Select columns containing ROI point values and convert as string np.array
  cnt = elem_df.iloc[1:, 1:6].values.astype(str)
  
  # Reshape from (N, 5) to (N*5, 1)
  cnt = cnt.reshape(cnt.shape[0] * cnt.shape[1], 1)
  
  # Remove rows with missing values
  cnt = cnt[~np.any(cnt == 'nan', axis=1)]
  
  # Split rows into x and y positions
  cnt = np.char.split(cnt, ',')

  # Convert from string to float
  cnt = np.array(cnt.tolist(), dtype=float)
  
  # Remove axes of length 1: reshape from (N, 1, 2) to (N, 2) 
  cnt = np.squeeze(cnt, axis=1)
  
  output = cnt
  
  return output

# %% scale_elem_cnt() ----

def scale_elem_cnt(elem_cnt: np.ndarray, 
                   metadata_dict: dict, 
                   to: str = 'PALM') -> np.ndarray:

  """Scale element contours to associated image.
  
  Args:
    elem_cnt (numpy.ndarray): element-specific contours as an array of shape (N, 2).
    metadata_dict (dict): metadata sub-dict produced by load_ome_tiff().
    to (str): either 'image' or 'PALM'.
  
  Returns:
    numpy.ndarray: element-specific contours scaled to associated image or to PALM RoboSoftware.

  """
    
  cnt = elem_cnt.copy()
  
  stage_pos = [metadata_dict['image']['position_x'], 
               metadata_dict['image']['position_y']]
  
  phys_size = [metadata_dict['image']['physical_size_x'], 
               metadata_dict['image']['physical_size_y']]
  
  img_size_x = metadata_dict['image']['size_x']
  img_size_y = metadata_dict['image']['size_y']
  
  objective = int(metadata_dict['objectives']['nominal_magnification'])
  
  if objective not in [10, 20]:
    
    raise NotImplementedError(f'Contour scaling not implemented for {objective}X objective yet.')

  # Compute offset
  if objective == 20:
    
    offset_x = (((img_size_x / img_size_y) -1) * 100)
    
    offset_y = - offset_x
  
  if objective == 10:
    
    offset_x = - ((img_size_x / img_size_y) * 100)
    
    offset_y = - (((img_size_x / img_size_y) * 100) / 2)
  
  ### Rescale to PALM RoboSoftware
  if to == 'PALM':
    
    # Adjust x position
    # 1) x = x - offset_x
    cnt[:, 0] = cnt[:, 0] - offset_x

    # 2) x = x / (1 + [(1 / image size x) * 100]) 
    cnt[:, 0] = cnt[:, 0] / (1 + ((1 / img_size_x) * 100))

    # 3) x = (image size x / 2) - x
    cnt[:, 0] = (img_size_x / 2) - cnt[:, 0]
    
    # Adjust y position  
    # 1) y = y - offset_y
    cnt[:, 1] = cnt[:, 1] - offset_y

    # 2) y = [(image size y / 2) - y] / (1 + [(1 / image size y) * 100])
    cnt[:, 1] = ((img_size_y / 2) - cnt[:, 1]) / (1 + ((1 / img_size_y) * 100)) 

    # Rescale ROIs to PALM size
    cnt = cnt * phys_size

    # Correct ROI position based on stage position
    # => Subtract stage x/y positions to ROI x/y positions 
    cnt = cnt - stage_pos
    
    # Round to 1 decimal place
    cnt = np.round(cnt, 1)
  
  ### Rescale to image  
  else:
    
    # Correct ROI position based on stage position
    # => Add stage x/y positions to ROI x/y positions 
    cnt = cnt + stage_pos
  
    # Rescale ROIs to physical size
    cnt = cnt / phys_size
  
    # Adjust x position
    # 1) x = (image size x / 2) - x
    cnt[:, 0] = (img_size_x / 2) - cnt[:, 0]
  
    # 2) x = x + [(x / image size x) * 100]
    cnt[:, 0] = cnt[:, 0] + ((cnt[:, 0] / img_size_x) * 100)
    
    # 3) x = x + offset_x
    cnt[:, 0] = cnt[:, 0] + offset_x
  
    # Adjust y position
    # 1) y = (image size y / 2) - (y + [(y / image size y) * 100])  
    cnt[:, 1] = (img_size_y / 2) - (cnt[:, 1] + ((cnt[:, 1] / img_size_y) * 100)) 
  
    # 2) y = y + offset_y
    cnt[:, 1] = cnt[:, 1] + offset_y
  
  output = cnt
    
  return output

# %% get_palm_elem() ----

def get_palm_elem(file: str, 
                  metadata_dict: dict) -> dict:

  """Extract PALM element properties and scale to associated image.
  
  Args:
    file (str): path to PALM .txt file.
    metadata_dict (dict): metadata sub-dict produced by load_ome_tiff().
  
  Returns:
    dict: dict of element-specific dicts. Each sub-dict contains metadata and element contours.

  """
    
  dfs = txt_to_elem_dfs(file=file)
  
  # Tidy metadata / element positions and return as dict
  output = {}
  for i in np.arange(len(dfs)):
    
    # Create element ID
    id = f"elem{str(i+1)}"
    
    # Extract metadata 
    tmp = get_elem_metadata(elem_df=dfs[i])
        
    # Clean element position
    tmp['palm_cnt'] = get_elem_cnt(elem_df=dfs[i])
    
    tmp['cnt'] = scale_elem_cnt(elem_cnt=tmp['palm_cnt'],
                                metadata_dict=metadata_dict,
                                to='image')
    
    output[id] = tmp
      
  return output
  
# %% format_elem_cnt() ----

def format_elem_cnt(elem_cnt: np.ndarray, 
                    color: str, 
                    id: int) -> pd.DataFrame:

  """Format element contours for PALM RoboSoftware.
  
  Args:
    elem_cnt (numpy.ndarray): element-specific contours as an array of shape (N, 2).
    color (str): color to assign to element.
    id (int): ID to assign to element.
  
  Returns:
    pandas.DataFrame: element-specific DataFrame matching PALM RoboSoftware format.

  """
    
  header = pd.DataFrame({'Type': ['', 'Freehand'],
                         'Color': ['', color],
                         'Thickness': ['', '2'],
                         'No': ['', str(id)],
                         'Laser function': ['', 'Cut'],
                         'CutShot': ['', '0,0'],
                         'Area': ['', '-'], 
                         'Z': ['', '-'], 
                         'Well': ['', 'manual'], 
                         'Objective': ['', '20X'],
                         'Comment': ['', '.'],
                         'Coordinates': ['', '']})
    
  cnt = np.apply_along_axis(func1d=lambda x: ','.join(x.astype(str)), 
                            axis=1, 
                            arr=elem_cnt)
  
  pad = 5 - (cnt.shape[0] % 5)
  
  cnt = np.pad(cnt, (0, pad), mode='empty')
  
  nrows = int(cnt.shape[0] / 5)
  
  cnt = cnt.reshape(nrows, 5) 
  
  cnt = cnt[~np.all(cnt == '', axis=1)]
  
  cnt = pd.DataFrame(cnt,
                     columns=['Color', 'Thickness', 'No', 'Laser function', 'CutShot'])
  
  cnt = cnt.dropna(how='all')
  
  output = pd.concat([header, cnt],
                     ignore_index=True)
  
  output['Type'] = output['Type'].fillna('.')
  
  for i in output.columns:
    output[i] = output[i].fillna('')
  
  return output

# %% make_elem_txt() ----

def make_elem_txt(data_dict: dict, 
                  metadata_dict: dict, 
                  colors: list = ['red', 'blue', 'yellow', 'green']) -> str:

  """Create string with PALM element properties.
  
  Args:
    data_dict (dict): dict of image data.
    metadata_dict (dict): dict of image metadata.
    colors (list): list of colors to assign to simple positive, double positive and ambiguous double positive elements, respectively.
  
  Returns:
    str: string with PALM element properties to be written to .txt file.

  """
  
  date_time = datetime.now().strftime(f'%d.%m.%Y\t%H:%M:%S')
  
  header = f'''PALMRobo Elements
Version:	V 4.9.0.0
Date, Time :	{date_time}

MICROMETER
Elements :\n
'''

  ch0_nm = metadata_dict['channels'][0]['name']
  ch1_nm = metadata_dict['channels'][1]['name']

  ch0_cnt_neg = data_dict[ch0_nm]['cnt']['neg']
  ch0_cnt_pos = data_dict[ch0_nm]['cnt']['pos']
  ch0_cnt_amb = data_dict[ch0_nm]['cnt']['amb']
  ch1_cnt_neg = data_dict[ch1_nm]['cnt']['neg']
  
  output = []
  total_length = 0
  start_id = 1
  for i, j in zip([ch0_cnt_neg, ch0_cnt_pos, ch0_cnt_amb, ch1_cnt_neg], colors):
    
    length = len(i) 
    
    if length > 0:
      
      tmp = [scale_elem_cnt(elem_cnt=k, metadata_dict=metadata_dict, to='PALM') for k in i]
    
      tmp = [format_elem_cnt(elem_cnt=k, color=j, id=str(l)) for k, l in zip(tmp, np.arange(start_id, start_id+length+1))]
      
      output.extend(tmp)
      
      total_length += length
      
      start_id = total_length+1
    
  output = pd.concat(output)
  
  output = output.to_csv(index=False, sep='\t')
  
  output = header + output + '\n\n\n'
  
  return output
