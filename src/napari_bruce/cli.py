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
from . import configuration

# %% launch_napari_with_plugin() ----

def _is_unicode_supported():
  
  enc = getattr(sys.stdout, 'encoding', None) or ''
  
  return 'utf' in enc.lower()

def spin_until(condition, message='Starting napari-bruce...', delay=0.08, timeout=None):
  
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
  
  env.setdefault('QT_SCALE_FACTOR', '1')

  cmd = ['napari', '--with', 'napari-bruce']

  try:
    
    proc = subprocess.Popen(cmd, env=env)
    
    def condition():
      
      if ready_file.exists():
        
        return True
      
      return proc.poll() is not None
    
    print('\n')
    
    spin_until(condition=condition, timeout=timeout)
    
    if ready_file.exists():
      
      print('\u2714 Program started\n')
    
    elif proc.poll() is not None:
      
      print(f'napari process exited with code {proc.returncode}\n')
    
    else:
      
      print('Timeout waiting for napari to signal ready')
    
    returncode = proc.wait()
    
    return returncode

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
      action='store_true',
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
      
      print('\nChecking TensorFlow / StarDist GPU status...')
      
      try:
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        import tensorflow as tf
      
      except Exception as e:
        
        print(f'\n🔴 TensorFlow could not be imported.\n{type(e).__name__}: {e}\n')
                
        return
      
      gpus = tf.config.list_physical_devices('GPU')
      
      if not gpus:
        
        print('\n🔴 No GPU visible to TensorFlow.\nStarDist will run on CPU.\n')
        
        return

      print(f'\n🟢 GPU(s) visible to TensorFlow: {gpus}')
    
      if sys.platform.startswith('win'):
      
        if shutil.which('ptxas.exe') is None:
        
          print("\nWARNING: 'ptxas.exe' not found in PATH.")
          print('CUDA Toolkit is likely not installed correctly; TensorFlow may fall back to CPU.')
          print("Install CUDA Toolkit 11.2 and add its 'bin' folder to PATH.")
        
        else:
          
          print('\n🟢 CUDA compiler (ptxas.exe) found in PATH.')
      
      try:
        
        with tf.device('/GPU:0'):
          
          a = tf.random.normal([2000, 2000])
          b = tf.random.normal([2000, 2000])
          start = time.time()
          c = tf.matmul(a, b)
          _ = c.numpy() 
          elapsed = time.time() - start
          
        print(f'\n🟢 GPU functional test succeeded (matmul time: {elapsed:.3f}s)')
          
        if elapsed > 2.0:
          
          print('\nGPU test ran but was slow.\nThis may indicate CPU fallback.')
        
        else:
          
          print('\n🟢 StarDist should run on GPU correctly.\n')
        
      except Exception as e:
        
        print(f'\n🔴 GPU functional test failed.\n{type(e).__name__}: {e}')
        print('TensorFlow detected a GPU but cannot execute on it.\nStarDist will fall back to CPU.\n')
      
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
    
    if args.version:
      
      print(r"""
        '            '
     /*/    '   '    \*\
   /**/     |\_/|     \**\
  *****-----*****-----*****
 |********* Bruce *********|
  ****/-\***********/-\****
   |*|   \*********/   |*|
    \*\    \*****/    /*/
      \\     \*/     //
        '     '     '
        """)
      
      print(f'napari-bruce {configuration.get_version()}\n')
      
      return
      
    configuration.get_config()
    
    if shutil.which('java') is None:
      
      raise RuntimeError('java not found on PATH; please install OpenJDK.')

    launch_napari_with_plugin()
    
  except Exception as e:
    
    print(f'{type(e).__name__}: {e}', file=sys.stderr)
    
    raise SystemExit(1)