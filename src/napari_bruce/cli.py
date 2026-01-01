# %% Set up ----

# Import required libraries
import os
import sys
import argparse
import subprocess
import napari_bruce.configuration as configuration

# %% launch_napari_with_plugin() ----

def launch_napari_with_plugin() -> None:
  
  """Start napari and load the napari-bruce plugin using the napari CLI."""
  
  cmd = ["napari", "--with", "napari-bruce"]
  
  try:
    
    subprocess.run(cmd, check=True)
  
  except FileNotFoundError:
    
    print("Error: 'napari' command not found.\nMake sure napari is installed in this environment and on your PATH.",
          file=sys.stderr)
    
    sys.exit(1)
  
  except subprocess.CalledProcessError as e:
    
    print(f'napari exited with error code {e.returncode}', 
          file=sys.stderr)
    
    sys.exit(e.returncode)

# %% main() ----

def main(argv: list[str] | None = None) -> None:
  
  parser = argparse.ArgumentParser(
    prog='bruce',
    description='Command-line interface for the napari-bruce plugin.',
    )
  
  parser.add_argument(
    '--show-config-path',
    action='store_true',
    help='print the path of the configuration file and exit.',
    )
  
  parser.add_argument(
    '--edit-config',
    action='store_true',
    help='open the configuration file in the default editor.',
    )
  
  parser.add_argument(
    '--reset-config',
    action='store_true',
    help='reset the configuration to defaults and exit.',
    )
  
  parser.add_argument(
    '--add-model',
    metavar='MODEL_DIR',
    help='add the StarDist model located at MODEL_DIR to napari-bruce.',
    )
  
  parser.add_argument(
    '--list-models',
    action='store_true',
    help='list available StarDist models.',
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
  
  if args.add_model is not None:
    
    configuration.add_stardist_model(args.add_model)
    print(f'Added StarDist model from:\n{args.add_model}')
    
    return
  
  if args.list_models:
    
    models = configuration.list_stardist_models()
    models = '\n- '.join(f'{k}: {v}' for k, v in models.items())
    print(f'Available StarDist models:\n- {models}')
    
    return
  
  launch_napari_with_plugin()

# %%
