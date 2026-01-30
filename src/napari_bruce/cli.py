# %% Set up ----

# Import required libraries
import os
import sys
import argparse
import subprocess
import shutil
import time
import threading
import itertools
import tempfile
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import napari_bruce.configuration as configuration

# %% launch_napari_with_plugin() ----

def _is_unicode_supported():
  
  enc = getattr(sys.stdout, 'encoding', None) or ''
  
  return 'utf' in enc.lower()

def spin_until(condition, message='Waiting…', delay=0.08, timeout=None):
  
  if not sys.stdout.isatty():
    
    start = time.time()
    
    while True:
      
      if condition():
        
        return True
      
      if timeout is not None and (time.time() - start) > timeout:
        
        return False
      
      time.sleep(0.05)

  stop_event = threading.Event()
  
  frames = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' if _is_unicode_supported() else '|/-\\'
  
  line_len = len(message) + 2

  def _spin():
    
    for f in itertools.cycle(frames):
      
      if stop_event.is_set():
        
        break
      
      sys.stdout.write(f"\r{f} {message}")
      
      sys.stdout.flush()
      
      time.sleep(delay)

  t = threading.Thread(target=_spin, daemon=True)
  
  t.start()

  start = time.time()
  
  try:
    
    while True:
      
      if condition():
        
        return True
      
      if timeout is not None and (time.time() - start) > timeout:
        
        return False
      
      time.sleep(0.05)
  
  finally:
    
    stop_event.set()
    
    t.join()
    
    sys.stdout.write('\r' + ' ' * line_len + '\r')
    
    sys.stdout.flush()

def launch_napari_with_plugin(timeout=60.0):
  
  ready_file = Path(os.path.join(tempfile.gettempdir(), 'napari_bruce_ready'))
  
  try:
    
    if ready_file.exists():
      
      ready_file.unlink()
  
  except Exception:
    
    pass

  env = os.environ.copy()
  
  env['NAPARI_BRUCE_READY_FILE'] = str(ready_file)

  print(r"""
        '            '
     /*/    '   '    \*\
   /**/     |\_/|     \**\
  *****-----*****-----*****
 |********* BRUCE *********|
  ****/-\***********/-\****
   |*|   \*********/   |*|
    \*\    \*****/    /*/
      \\     \*/     //
        '     '     '
        """)

  cmd = ['napari', '--with', 'napari-bruce']

  try:
    
    proc = subprocess.Popen(cmd, env=env)
    
    def condition():
      
      if ready_file.exists():
        
        return True
      
      return proc.poll() is not None

    ok = spin_until(condition, message='Starting napari-bruce...', timeout=timeout)
    
    if ready_file.exists():
      
      print('✔ Program is ready\n')
    
    elif proc.poll() is not None:
      
      print(f'napari process exited with code {proc.returncode}\n')
    
    else:
      
      print('Timeout waiting for napari to signal ready')

  except FileNotFoundError:
    
    print("FileNotFoundError: 'napari' not found. Ensure it is installed and on PATH.\n", file=sys.stderr)
    
    raise SystemExit(1)

# %% cli_main() ----

def cli_main(argv: list[str] | None = None) -> None:
  
  try:
    
    parser = argparse.ArgumentParser(
      prog='bruce',
      description='Command-line interface for the napari-bruce plugin.',
      )
  
    parser.add_argument(
      '--show-config-path',
      action='store_true',
      help='print the path of the configuration file and exit',
      )
  
    parser.add_argument(
      '--edit-config',
      action='store_true',
      help='open the configuration file in the default editor',
      )
  
    parser.add_argument(
      '--reset-config',
      action='store_true',
      help='reset the configuration to defaults and exit',
      )
    
    parser.add_argument(
      '--gpu-status',
      action='store_true',
      help='check if GPU(s) are visible to TensorFlow',
      )
    
    parser.add_argument(
      '--list-models',
      action='store_true',
      help='list available StarDist models',
      )
  
    parser.add_argument(
      '--add-model',
      metavar='MODEL_DIR',
      help='add the StarDist model located at MODEL_DIR to napari-bruce',
      )
    
    parser.add_argument(
      '-v', '--version',
      action="version",
      version=f'napari-bruce {configuration.get_version()}',
      help='show the napari-bruce version and exit',
      )
  
    args = parser.parse_args(argv)
  
    if args.show_config_path:
    
      config_path = configuration.get_config_file_path()
      if not os.path.exists(config_path):
        configuration.make_default_config()
      print(config_path)
    
      return
  
    if args.edit_config:
    
      config_path = configuration.get_config_file_path()
      if not os.path.exists(config_path):
        configuration.make_default_config()
      print(f'Opening config file at:\n{config_path}')
      configuration.open_in_editor(config_path)
    
      return
  
    if args.reset_config:
    
      config_path = configuration.get_config_file_path()
      configuration.make_default_config()
      print(f'Configuration reset to defaults at:\n{config_path}')
    
      return
    
    if args.gpu_status:
      
      with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
        
        from tensorflow.config import list_physical_devices
        gpus = list_physical_devices('GPU')
      
      if gpus: 
        print(f'🟢 StarDist runs on GPU.\nGPU(s) visible to TensorFlow: {gpus}')
      else:
        print('🔴 Stardist runs on CPU.')
      
      return
    
    if args.list_models:
    
      models = configuration.list_stardist_models()
      models = '\n- '.join(f'{k}: {v}' for k, v in models.items())
      print(f'Available StarDist models:\n- {models}')
    
      return
  
    if args.add_model is not None:
    
      configuration.add_stardist_model(args.add_model)
      print(f'Added StarDist model from:\n{args.add_model}')
    
      return
  
    configuration.get_config()
    
    if shutil.which('java') is None:
      
      raise RuntimeError('java not found on PATH; please install OpenJDK.')

    launch_napari_with_plugin()
    
  except Exception as e:
    
    print(f'{type(e).__name__}: {e}', file=sys.stderr)
    
    raise SystemExit(1)

# %%
