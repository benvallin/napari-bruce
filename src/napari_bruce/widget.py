# %% Set up ----

print(f'\nStarting napari-bruce 🦇...\n')
    
# Import required libraries
print(f'\n\t⏳ Loading dependencies\n')

import os
import copy
import numpy as np
import cv2
import json
import importlib.resources
from matplotlib.colors import to_rgba
from pathlib import Path
from qtpy.QtWidgets import QPushButton, QWidget, QLabel, QVBoxLayout, QFileDialog, QDoubleSpinBox, QHBoxLayout, QApplication
from qtpy.QtCore import Signal, QObject, QThread, QTimer
from csbdeep.utils import normalize
from contextlib import redirect_stdout, redirect_stderr
import napari_bruce.configuration as configuration
import napari_bruce.workflow as workflow

# Check Java
workflow.require_java()

# Load configuration
config = configuration.get_config()

# Convert config['channels'] keys to int if read from json file 
config['channels'] = {int(k):v for k, v in config['channels'].items()}
              
# Import StarDist and define models 
print(f'\t⏳ Loading StarDist models\n')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
  
  from stardist.models import StarDist2D
  
  pretrained = [k for k, v in configuration.list_stardist_models().items() if v == 'pretrained']
  
  models = {}
  
  for i in [0, 1]:
    
    model_nm = config['channels'][i]['stardist_model']
    
    if model_nm in pretrained:
      
      models[i] = StarDist2D.from_pretrained(model_nm)
    
    else:
      
      stardist_models_dir_path = os.path.join(importlib.resources.files('napari_bruce'), 
                                              'stardist_models')
      
      models[i] = StarDist2D(None, 
                             name=model_nm, 
                             basedir=stardist_models_dir_path)
  
print(f'\n*** 🟢 Program is ready ***\n')

# %% ParamValueBox() ----

class ParamValueBox(QWidget):
  
  valueChanged = Signal(float)
  
  def __init__(self, 
               label, 
               default,
               min_val, 
               max_val, 
               decimals=0, 
               parent=None):
    
    super().__init__(parent)
    
    layout = QHBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    
    self._label = QLabel(label)
    self._spin = QDoubleSpinBox()
    self._spin.setDecimals(decimals)
    self._spin.setRange(min_val, max_val)
    self._spin.setValue(default)
    
    layout.addWidget(self._label)
    layout.addWidget(self._spin)
    
    self.setFixedHeight(QPushButton().sizeHint().height())

    self._spin.valueChanged.connect(self.valueChanged.emit)

  def value(self) -> float:
    return self._spin.value()

# %% PluginManager() ----

class PluginManager(QWidget):
    
  def __init__(self, 
               napari_viewer: 'napari.viewer.Viewer'):
    
    super().__init__()
    
    # Define attributes 
    self.viewer = napari_viewer    
    
    self.config = copy.deepcopy(config)
    
    for i in ['layers', 'path', 'data', 'metadata', 'ch_names',
              '_load_worker_thread', '_load_worker',
              '_predict_worker_thread', '_predict_worker',
              '_filter_size_worker_thread', '_filter_size_worker',
              '_apply_edits_worker_thread', '_apply_edits_worker',
              '_overlap_worker_thread', '_overlap_worker',
              'btn_clear',
              'btn_load',
              'btn_predict',
              'btn_filter_size',
              'box_min_n_pix_ch0',
              'box_min_n_pix_ch1',
              'btn_apply_edits',
              'btn_overlap',
              'btn_filter_overlap',
              'box_min_pct_ovl_ch0_by_ch1',
              'box_min_pct_ovl_ch1_by_ch0',
              'btn_save']:
      
      setattr(self, i, None)
      
    for i in ['_predict_requested', 
              '_filter_size_requested',
              '_filter_overlap_requested']:
      
      setattr(self, i, False)  
      
    self._msg = '' 
    
    self.btn_select = QPushButton('Select file')
    self.btn_select.clicked.connect(self.on_select_clicked)
    
    self.lab_message = QLabel('')
    
    # Create default layout    
    self.header_layout = QVBoxLayout()  
    self.header_layout.addWidget(self.btn_select)
    self.header_layout.addSpacing(30)
    self.header_layout.addWidget(self.lab_message)
    self.layout = QVBoxLayout()
    self.layout.addLayout(self.header_layout)
    self.layout.addStretch(1)  
    self.setLayout(self.layout) 
    self.setMinimumWidth(300)
    self.setMaximumWidth(1000)   
    
  # Define methods  
  def send_message(self, text):
    
    self.lab_message.setText(text)
    self.lab_message.setWordWrap(True)
    
  def on_clear_clicked(self, msg: str = '', *args):
    
    msg = '' if msg in (None, False, True) else str(msg)
    
    if msg == '':    
      self.send_message('⏳ Resetting napari-bruce...')
    
    # Reset attributes to default values
    self.config = copy.deepcopy(config)
    
    for i in ['btn_clear',
              'btn_load',
              'btn_predict',
              'btn_filter_size',
              'box_min_n_pix_ch0',
              'box_min_n_pix_ch1',
              'btn_apply_edits',
              'btn_overlap',
              'btn_filter_overlap',
              'box_min_pct_ovl_ch0_by_ch1',
              'box_min_pct_ovl_ch1_by_ch0',
              'btn_save']:
      
      j = getattr(self, i, None)
      
      if j is not None:
        
        self.layout.removeWidget(j)
        j.hide()
        j.deleteLater()    
        setattr(self, i, None)
    
    for i in ['layers', 
              'path', 
              'data', 
              'metadata', 
              'ch_names']:
      
      setattr(self, i, None)
    
    for i in ['_predict_requested', 
              '_filter_size_requested',
              '_filter_overlap_requested']:
      
      setattr(self, i, False)  
    
    self._msg = '' 
    
    # Delay viewer reset
    QApplication.processEvents()
    
    QTimer.singleShot(50, lambda: self._reset_viewer_to_defaults(msg))
  
  def _reset_viewer_to_defaults(self, msg: str = ''):
    
    # Clear all layers 
    self.viewer.layers.clear()
    
    # Recreate 'Select file' button 
    self.btn_select = QPushButton('Select file')
    self.btn_select.clicked.connect(self.on_select_clicked)
    self.layout.insertWidget(0, self.btn_select)
    
    # Send clear message
    self.send_message(msg)
       
  def on_select_clicked(self):
    
    # Record user-selected file path
    self.path = QFileDialog.getOpenFileName(parent=self, 
                                            caption='Select .zvi file',
                                            directory=str(Path(self.config['in_dir_path']).expanduser()),
                                            filter='Images (*.zvi)')[0]
    
    # Abort if user cancelled selection
    if self.path == '':
      
      self.send_message(f'{self._msg}.zvi file selection cancelled')
      
      return
    
    # Remove 'Select file' button from viewer
    self.layout.removeWidget(self.btn_select)
    self.btn_select.hide()
    self.btn_select.deleteLater()    
    self.btn_select = None
        
    # Add 'Load images', 'Predict cells' and 'Clear' buttons to viewer
    self.btn_load = QPushButton('Load images')
    self.btn_load.clicked.connect(self.on_load_clicked)
    
    self.btn_predict = QPushButton('Predict cells')
    self.btn_predict.clicked.connect(self.on_predict_clicked)

    self.btn_clear = QPushButton('Clear')
    self.btn_clear.clicked.connect(lambda checked=False: self.on_clear_clicked(''))
    
    for i, j in zip([0, 1, 2],
                    [self.btn_load,
                     self.btn_predict, 
                     self.btn_clear]):
      
      self.layout.insertWidget(i, j)
    
    self.send_message(f'{self._msg}👍 .zvi file selected')
  
  def on_load_clicked(self):
    
    # Remove 'Load images' button from viewer
    self.layout.removeWidget(self.btn_load)
    self.btn_load.hide()
    self.btn_load.deleteLater()    
    self.btn_load = None
        
    self.send_message(f'{self._msg}💪 Loading images...')
    
    # Trigger load worker thread 
    self.start_load_thread()
          
  def start_load_thread(self):
    
    # Disable 'Predict cells' and 'Clear' buttons
    # => 'Predict cells' button does not exist if clicked directly
    for i in [self.btn_predict,
              self.btn_clear]:
      
      if i is not None:
        
        i.setEnabled(False)
      
    # Create thread and worker
    self._load_worker_thread = QThread()
    self._load_worker = LoadWorker(self.path, self.config)
    self._load_worker.moveToThread(self._load_worker_thread)
    
    # Start worker when thread starts
    self._load_worker_thread.started.connect(self._load_worker.run)

    # Ensure UI messages and result handling
    self._load_worker.sig_message.connect(self.send_message)
    self._load_worker.sig_output.connect(self.on_load_finished)

    # Make thread stop when worker emits sig_output
    self._load_worker.sig_output.connect(self._load_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._load_worker_thread.finished.connect(self._load_worker.deleteLater)

    # Delete thread itself on finished 
    self._load_worker_thread.finished.connect(self._load_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def _reset_attrs():
      
      for i in ['_load_worker_thread', '_load_worker']:
        
        setattr(self, i, None)
        
    self._load_worker_thread.finished.connect(_reset_attrs)

    # Start worker thread
    self._load_worker_thread.start()

  def on_load_finished(self, data, metadata):
        
    # Warn if channel names are not consistent between metadata and config 
    ch0_nm_ok = metadata['channels'][0]['name'] == self.config['channels'][0]['name']
    ch1_nm_ok = metadata['channels'][1]['name'] == self.config['channels'][1]['name']
    
    if not (ch0_nm_ok and ch1_nm_ok):
      
      self._msg = f'{self._msg}⚠️ Channel name mismatch between image and config\n\n'
    
    # Update data / metadata / ch_names attributes with load worker output
    self.data = data
    self.metadata = metadata
    self.ch_names = {}
    for i in [0, 1]:
      self.ch_names[i] = self.metadata['channels'][i]['name']
    
    # Delay viewer update
    self.send_message(f'{self._msg}⏳ Updating viewer...')
    
    QApplication.processEvents()
    
    QTimer.singleShot(50, self._update_viewer_on_load_finished)
    
  def _update_viewer_on_load_finished(self):
      
    # Add normalized images to viewer
    for i in self.ch_names.values():
      
      self.viewer.add_image(self.data[i]['norm_img'], name=f'{i} normalized image')
    
    # Add empty labels, shapes and merge image layers to viewer
    self.layers = {}
    
    for k, v in self.ch_names.items():  
      
      self.layers[v] = {}
      
      temp_array = np.zeros(self.data[v]['norm_img'].shape, 
                            dtype=int)
      
      color_dict = {0: (0.0, 0.0, 0.0, 0.0),
                    None: to_rgba(self.config['channels'][k]['color'])}
      
      self.layers[v]['labels_layer'] = self.viewer.add_labels(temp_array,
                                                              name=f'{v} - remove',
                                                              opacity=1.0,
                                                              colormap=color_dict,
                                                              visible=True)
      self.layers[v]['labels_layer'].visible = False
      self.layers[v]['labels_layer'].contour = 8
      self.layers[v]['labels_layer'].brush_size = 80
      
      self.layers[v]['shapes_layer'] = self.viewer.add_shapes(data = [],
                                                              name = f'{v} - add',
                                                              shape_type = 'path',
                                                              edge_color=self.config['channels'][k]['color'],
                                                              edge_width=6,
                                                              visible=True)
      self.layers[v]['shapes_layer'].mode = 'add_path'
      self.layers[v]['shapes_layer'].visible = False

    temp_array = np.zeros(self.data[self.ch_names[0]]['norm_img'].shape+(3,), 
                          dtype=np.uint8)
    
    self.layers['merge'] = self.viewer.add_image(temp_array, 
                                                 name=f'Merge + ROIs',
                                                 rgb=True,
                                                 visible=False)
    
    # Set active layer to channel 1 normalized image
    self.viewer.layers.selection.active = self.viewer.layers[f'{self.ch_names[1]} normalized image']
    
    # Add 'min n pix' boxes to viewer
    self.box_min_n_pix_ch0 = ParamValueBox(label=f'min n pix {self.ch_names[0]}:', 
                                           default=self.config['channels'][0]['min_n_pix'],
                                           min_val=0.0, 
                                           max_val=10000.0)
    self.box_min_n_pix_ch0.valueChanged.connect(self.on_min_n_pix_ch0_changed)
      
    self.box_min_n_pix_ch1 = ParamValueBox(label=f'min n pix {self.ch_names[1]}:', 
                                           default=self.config['channels'][1]['min_n_pix'],
                                           min_val=0.0, 
                                           max_val=10000.0)
    self.box_min_n_pix_ch1.valueChanged.connect(self.on_min_n_pix_ch1_changed)
      
    for i, j in zip([0, 1], 
                    [self.box_min_n_pix_ch0, 
                     self.box_min_n_pix_ch1]):
      
      self.layout.insertWidget(i, j)
    
    # Stop here if 'Load images' button was clicked
    if not self._predict_requested:
      
      # Re-enable 'Predict cells' and 'Clear' buttons
      for i in [self.btn_predict, 
                self.btn_clear]:
        
        i.setEnabled(True)
        
      self.send_message(f"{self._msg}🔎 Check images in viewer\n\n🔹 Adjust min size thresholds (optional)\n\n👉 Press 'Predict cells' when ready")
          
    # Alternatively, proceed with prediction if 'Predict cells' button was clicked
    else:
      
      # Disable 'min n pix' boxes
      for i in [self.box_min_n_pix_ch0,  
                self.box_min_n_pix_ch1]:
        
        i.setEnabled(False)
      
      # Proceed with prediction
      self.on_img_loaded_and_predict_clicked()
      
  def on_predict_clicked(self):
    
    # Remove 'Load images' and 'Predict cells' buttons from viewer 
    # => If 'Load images' button was clicked previously, it has been deleted already 
    for i in ['btn_load',
              'btn_predict']:
      
      j = getattr(self, i, None)
      
      if j is not None:
        
        self.layout.removeWidget(j)
        j.hide()
        j.deleteLater()    
        setattr(self, i, None)
    
    # Trigger load worker thread if 'Predict cells' button was clicked directly
    if self.data is None:
      
      # Record that prediction has been requested
      self._predict_requested = True
      
      self.send_message(f'{self._msg}💪 Loading images...')
      
      # Trigger load worker thread 
      self.start_load_thread()
      
      return
    
    # Alternatively, proceed with prediction if 'Load images' button was clicked previously
    self.on_img_loaded_and_predict_clicked()
  
  def on_img_loaded_and_predict_clicked(self):
          
    self.send_message(f'{self._msg}💪 Predicting objects...')
    
    # Trigger predict worker thread 
    self.start_predict_thread()
          
  def start_predict_thread(self):
    
    # Disable 'min n pix' boxes and 'Clear' button
    for i in [self.box_min_n_pix_ch0, 
              self.box_min_n_pix_ch1,
              self.btn_clear]:
      
      i.setEnabled(False)
            
    # Create thread and worker
    self._predict_worker_thread = QThread()
    self._predict_worker = PredictWorker(self.data, self.ch_names)
    self._predict_worker.moveToThread(self._predict_worker_thread)
    
    # Start worker when thread starts
    self._predict_worker_thread.started.connect(self._predict_worker.run)

    # Ensure UI messages and result handling
    self._predict_worker.sig_message.connect(self.send_message)
    self._predict_worker.sig_output.connect(self.on_predict_finished)

    # Make thread stop when worker emits sig_output
    self._predict_worker.sig_output.connect(self._predict_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._predict_worker_thread.finished.connect(self._predict_worker.deleteLater)

    # Delete thread itself on finished 
    self._predict_worker_thread.finished.connect(self._predict_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def _reset_attrs():
      
      for i in ['_predict_worker_thread', '_predict_worker']:
        
        setattr(self, i, None)
        
    self._predict_worker_thread.finished.connect(_reset_attrs)

    # Start worker thread
    self._predict_worker_thread.start()
    
  def on_predict_finished(self, data):
    
    # Update data attribute with predict worker output
    self.data = data
    
    self.send_message(f'{self._msg}💪 Filtering out small objects...')
    
    # Trigger filter size worker thread 
    self.start_filter_size_thread()    
    
  def on_filter_size_clicked(self):
    
    # Record that size filtering has been requested
    self._filter_size_requested = True
    
    self.send_message(f'{self._msg}💪 Re-filtering out small objects...')
    
    # Trigger filter size worker thread 
    self.start_filter_size_thread()
            
  def start_filter_size_thread(self):
    
    # Disable 'min n pix boxes', 'Adjust size filter', 'Apply edits' and 'Clear' buttons
    # => 'Adjust size filter' and 'Apply edits' buttons do not exist yet if predictions are returned for the first time 
    for i in [self.box_min_n_pix_ch0, 
              self.box_min_n_pix_ch1,
              self.btn_filter_size,
              self.btn_apply_edits,
              self.btn_clear]:
      
      if i is not None:
        
        i.setEnabled(False)
        
    # Create thread and worker
    self._filter_size_worker_thread = QThread()
    self._filter_size_worker = FilterSizeWorker(self.data, self.config, self.ch_names)
    self._filter_size_worker.moveToThread(self._filter_size_worker_thread)
    
    # Start worker when thread starts
    self._filter_size_worker_thread.started.connect(self._filter_size_worker.run)

    # Ensure UI messages and result handling
    self._filter_size_worker.sig_message.connect(self.send_message)
    self._filter_size_worker.sig_output.connect(self.on_filter_size_finished)

    # Make thread stop when worker emits sig_output
    self._filter_size_worker.sig_output.connect(self._filter_size_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._filter_size_worker_thread.finished.connect(self._filter_size_worker.deleteLater)

    # Delete thread itself on finished 
    self._filter_size_worker_thread.finished.connect(self._filter_size_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def _reset_attrs():
      
      for i in ['_filter_size_worker_thread', '_filter_size_worker']:
        
        setattr(self, i, None)
        
    self._filter_size_worker_thread.finished.connect(_reset_attrs)

    # Start worker thread
    self._filter_size_worker_thread.start()
    
  def on_filter_size_finished(self, data):    
    
    # Update data attribute with filter size worker output
    self.data = data
    
    # Delay viewer update
    self.send_message(f'{self._msg}⏳ Updating viewer...')
    
    QApplication.processEvents()
    
    QTimer.singleShot(50, self._update_viewer_on_filter_size_finished)
  
  def _update_viewer_on_filter_size_finished(self):
    
    # Update labels and shapes layers 
    for v in self.ch_names.values():
      
      self.layers[v]['labels_layer'].visible = False
      self.layers[v]['labels_layer'].data = self.data[v]['filt_msk_edited']       
      self.layers[v]['labels_layer'].visible = True    
      self.layers[v]['labels_layer'].mode = 'erase'
      
      self.layers[v]['shapes_layer'].visible = True 
    
    # Set active layer to channel 1 shapes layer
    self.viewer.layers.selection.active = self.layers[self.ch_names[1]]['shapes_layer']
    
    # Add 'Adjust size filter' and 'Apply edits' buttons to viewer if predictions are returned for the first time 
    if not self._filter_size_requested:
      
      self.btn_filter_size = QPushButton('Adjust size filter')
      self.btn_filter_size.clicked.connect(self.on_filter_size_clicked)
      
      self.btn_apply_edits = QPushButton('Apply edits')
      self.btn_apply_edits.clicked.connect(self.on_apply_edits_clicked)
      
      for i, j in zip([2, 3],
                      [self.btn_filter_size,
                       self.btn_apply_edits]):
        
        self.layout.insertWidget(i, j)
    
    # Re-enable 'min n pix' boxes, 'Adjust size filter', 'Apply edits' and 'Clear' buttons
    # => 'Adjust size filter' and 'Apply edits' buttons are already enabled if predictions are returned for the first time 
    for i in [self.box_min_n_pix_ch0, 
              self.box_min_n_pix_ch1,
              self.btn_filter_size,
              self.btn_apply_edits,
              self.btn_clear]:
      
      i.setEnabled(True)
        
    self.send_message(f"{self._msg}🔎 Check images in viewer\n\n🔹 Edit cell contours (optional)\n\n👉 Press 'Apply edits' when finished")
        
  def on_apply_edits_clicked(self):
    
    # Remove 'min n pix' boxes, 'Adjust size filter' and 'Apply edits' buttons from viewer
    for i in ['box_min_n_pix_ch0',
              'box_min_n_pix_ch1',
              'btn_filter_size',
              'btn_apply_edits']:
      
      j = getattr(self, i, None)
      self.layout.removeWidget(j)
      j.hide()
      j.deleteLater()    
      setattr(self, i, None)
            
    self.send_message(f'{self._msg}💪 Applying edits...')
    
    # Trigger apply edits worker thread 
    self.start_apply_edits_thread()
      
  def start_apply_edits_thread(self):
    
    # Disable 'Clear' button
    self.btn_clear.setEnabled(False)
    
    # Create thread and worker
    self._apply_edits_worker_thread = QThread()
    self._apply_edits_worker = ApplyEditsWorker(self.layers, self.data, self.config, self.ch_names)
    self._apply_edits_worker.moveToThread(self._apply_edits_worker_thread)
    
    # Start worker when thread starts
    self._apply_edits_worker_thread.started.connect(self._apply_edits_worker.run)

    # Ensure UI messages and result handling
    self._apply_edits_worker.sig_message.connect(self.send_message)
    self._apply_edits_worker.sig_output.connect(self.on_apply_edits_finished)

    # Make thread stop when worker emits sig_output
    self._apply_edits_worker.sig_output.connect(self._apply_edits_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._apply_edits_worker_thread.finished.connect(self._apply_edits_worker.deleteLater)

    # Delete thread itself on finished 
    self._apply_edits_worker_thread.finished.connect(self._apply_edits_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def _reset_attrs():
      
      for i in ['_apply_edits_worker_thread', '_apply_edits_worker']:
        
        setattr(self, i, None)
        
    self._apply_edits_worker_thread.finished.connect(_reset_attrs)

    # Start worker thread
    self._apply_edits_worker_thread.start()
    
  def on_apply_edits_finished(self, data):
    
    # Update data attribute with apply edits worker output
    self.data = data
    
    # Delay viewer update
    self.send_message(f'{self._msg}⏳ Updating viewer...')
    
    QApplication.processEvents()
    
    QTimer.singleShot(50, self._update_viewer_on_apply_edits_finished)
  
  def _update_viewer_on_apply_edits_finished(self):
    
    # Remove shapes and labels layers 
    for v in self.ch_names.values():
      
      self.viewer.layers.remove(self.layers[v]['shapes_layer'])
      self.viewer.layers.remove(self.layers[v]['labels_layer'])
      
    # Update merge image layer
    self.layers['merge'].visible = False
    
    self.layers['merge'].data = self.data['merge']['merge_norm_img_rois']
    
    self.layers['merge'].visible = True
    
    # Set active layer to merge image layer
    self.viewer.layers.selection.active = self.layers['merge']

    # Add 'min % ovl' boxes and 'Find overlaps' button to viewer 
    self.box_min_pct_ovl_ch0_by_ch1 = ParamValueBox(label=f'min %ovl {self.ch_names[0]}/{self.ch_names[1]}:', 
                                                    default=self.config['min_pct_ovl_ch0_by_ch1'],
                                                    min_val=0.0, 
                                                    max_val=100.0)
    self.box_min_pct_ovl_ch0_by_ch1.valueChanged.connect(self.on_min_pct_ovl_ch0_by_ch1_changed)
    
    self.box_min_pct_ovl_ch1_by_ch0 = ParamValueBox(label=f'min %ovl {self.ch_names[1]}/{self.ch_names[0]}:', 
                                                    default=self.config['min_pct_ovl_ch1_by_ch0'],
                                                    min_val=0.0, 
                                                    max_val=100.0)
    self.box_min_pct_ovl_ch1_by_ch0.valueChanged.connect(self.on_min_pct_ovl_ch1_by_ch0_changed)
    
    self.btn_overlap = QPushButton('Find overlaps')
    self.btn_overlap.clicked.connect(self.on_overlap_clicked)
    
    for i, j in zip([0, 1, 2],
                    [self.box_min_pct_ovl_ch0_by_ch1,
                     self.box_min_pct_ovl_ch1_by_ch0, 
                     self.btn_overlap]):
      
      self.layout.insertWidget(i, j)
    
    # Re-enable 'Clear' button
    self.btn_clear.setEnabled(True)
    
    self.send_message(f"{self._msg}🔹 Adjust overlap thresholds (optional)\n\n👉 Press 'Find overlaps' when ready")
       
  def on_overlap_clicked(self):
    
    # Remove 'Find overlaps' button from viewer
    self.layout.removeWidget(self.btn_overlap)
    self.btn_overlap.hide()
    self.btn_overlap.deleteLater()    
    self.btn_overlap = None
        
    self.send_message(f'{self._msg}💪 Computing overlaps...')
    
    # Trigger overlap worker thread   
    self.start_overlap_thread()
      
  def on_filter_overlap_clicked(self):
    
    # Record that % overlap filtering has been requested
    self._filter_overlap_requested = True
    
    self.send_message(f'{self._msg}💪 Re-computing overlaps...')
    
    # Trigger overlap worker thread   
    self.start_overlap_thread()
          
  def start_overlap_thread(self):
    
    # Disable 'min % ovl' boxes, 'Adjust overlap filter', 'Save results' and 'Clear' buttons
    # => 'Adjust overlap filter' and 'Save results' buttons do not exist yet if overlaps are returned for the first time 
    for i in [self.box_min_pct_ovl_ch0_by_ch1,
              self.box_min_pct_ovl_ch1_by_ch0, 
              self.btn_filter_overlap,
              self.btn_save,
              self.btn_clear]:
      
      if i is not None:
        
        i.setEnabled(False)
    
    # Create thread and worker
    self._overlap_worker_thread = QThread()
    self._overlap_worker = OverlapWorker(self.data, self.config, self.ch_names)
    self._overlap_worker.moveToThread(self._overlap_worker_thread)
    
    # Start worker when thread starts
    self._overlap_worker_thread.started.connect(self._overlap_worker.run)

    # Ensure UI messages and result handling
    self._overlap_worker.sig_message.connect(self.send_message)
    self._overlap_worker.sig_output.connect(self.on_overlap_finished)

    # Make thread stop when worker emits sig_output
    self._overlap_worker.sig_output.connect(self._overlap_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._overlap_worker_thread.finished.connect(self._overlap_worker.deleteLater)

    # Delete thread itself on finished 
    self._overlap_worker_thread.finished.connect(self._overlap_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def _reset_attrs():
      
      for i in ['_overlap_worker_thread', '_overlap_worker']:
        
        setattr(self, i, None)
        
    self._overlap_worker_thread.finished.connect(_reset_attrs)

    # Start worker thread
    self._overlap_worker_thread.start()
    
  def on_overlap_finished(self, data):
    
    # Update data attribute with overlap worker output
    self.data = data
    
    # Delay viewer update
    self.send_message(f'{self._msg}⏳ Updating viewer...')
    
    QApplication.processEvents()
    
    QTimer.singleShot(50, self._update_viewer_on_overlap_finished)
  
  def _update_viewer_on_overlap_finished(self):
    
    # Update merge image layer
    self.layers['merge'].visible = False
    
    self.layers['merge'].data = self.data['merge']['merge_norm_img_status']
    
    self.layers['merge'].visible = True
    
    # Add 'Adjust overlap filter' and 'Save results' buttons to viewer if overlaps are returned for the first time
    if not self._filter_overlap_requested:
      
      self.btn_filter_overlap = QPushButton('Adjust overlap filter')
      self.btn_filter_overlap.clicked.connect(self.on_filter_overlap_clicked)
      
      self.btn_save = QPushButton('Save results')
      self.btn_save.clicked.connect(self.on_save_clicked)
      
      for i, j in zip([2, 3],
                      [self.btn_filter_overlap,
                       self.btn_save]):
        
        self.layout.insertWidget(i, j)
        
    # Re-enable 'min % ovl' boxes, 'Adjust overlap filter', 'Save results' and 'Clear' buttons
    # => 'Adjust overlap filter' and 'Save results' buttons do not exist yet if overlaps are returned for the first time 
    for i in [self.box_min_pct_ovl_ch0_by_ch1,
              self.box_min_pct_ovl_ch1_by_ch0, 
              self.btn_filter_overlap,
              self.btn_save,
              self.btn_clear]:
      
      i.setEnabled(True)
    
    # Message cell status summary 
    msg = f'''
    
    📚 Count summary
    
    
    🔹 {self.ch_names[0]}
    
      🔸 total: {self.data[self.ch_names[0]]['summary']['total']}
      🔸 {self.ch_names[1]}-neg: {self.data[self.ch_names[0]]['summary']['neg']}
      🔸 {self.ch_names[1]}-pos: {self.data[self.ch_names[0]]['summary']['pos']}
      🔸 {self.ch_names[1]}-ambiguous: {self.data[self.ch_names[0]]['summary']['amb']}
    
    
    🔹 {self.ch_names[1]} 
    
      🔸 total: {self.data[self.ch_names[1]]['summary']['total']}
      🔸 {self.ch_names[0]}-neg: {self.data[self.ch_names[1]]['summary']['neg']}
      🔸 {self.ch_names[0]}-pos: {self.data[self.ch_names[1]]['summary']['pos']}
      🔸 {self.ch_names[0]}-ambiguous: {self.data[self.ch_names[1]]['summary']['amb']}
    
    
    🔎 See merge image in viewer
    '''
    
    self.send_message(f'{self._msg}{msg}')
  
  def on_save_clicked(self):
    
    # Define output directory path 
    out_dir_path = os.path.join(Path(self.config['out_dir_path']).expanduser(), 
                                self.metadata['img_nm'])
    
    # Construct element list and write to file 
    self.elem_list = workflow.make_elem_txt(data_dict=self.data, metadata_dict=self.metadata)
    
    with open(os.path.join(out_dir_path, 'elem_list.txt'), 'w') as f:
      f.write(self.elem_list)
      
    # Write data and metadata to file
    workflow.pickle_data(data=self.data, filename=os.path.join(out_dir_path, 'data.pkl'))
    
    workflow.pickle_data(data=self.metadata, filename=os.path.join(out_dir_path, 'metadata.pkl'))
    
    # Write config to file
    with open(os.path.join(out_dir_path, 'config.json'), 'w') as f:
      json.dump(self.config, f, indent=2)
    
    self.send_message(f"{self._msg}💾 Results save at:\n{self.config['out_dir_path']}\n\nIn subfolder:\n{self.metadata['img_nm']}")

  def on_min_n_pix_ch0_changed(self, value):

    self.config['channels'][0]['min_n_pix'] = float(value)
    
  def on_min_n_pix_ch1_changed(self, value):

    self.config['channels'][1]['min_n_pix'] = float(value)
  
  def on_min_pct_ovl_ch0_by_ch1_changed(self, value):

    self.config['min_pct_ovl_ch0_by_ch1'] = float(value)
    
  def on_min_pct_ovl_ch1_by_ch0_changed(self, value):

    self.config['min_pct_ovl_ch1_by_ch0'] = float(value)
  
# %% LoadWorker() ----

class LoadWorker(QObject):
  
  sig_message = Signal(str)                  
  sig_output = Signal(object, object)
  
  def __init__(self, 
               path: str, 
               config: object,
               parent=None):
    
    super().__init__(parent)
    self.path = path
    self.config = config
    self.data = None
    self.metadata = None
    
  def run(self):
    
    # Define output paths 
    out_dir_path = os.path.join(Path(self.config['out_dir_path']).expanduser(), 
                                Path(self.path).stem)
    
    ome_tiff_file_path = os.path.join(out_dir_path, Path(self.path).stem+'.ome.tiff')
    
    # Convert PALM .zvi file to OME-TIFF
    workflow.convert_zvi_to_ome(file=self.path, 
                                out_dir_path=out_dir_path,
                                jar_pkg='napari_bruce.bioformats',
                                jar_name='bioformats_package.jar')
    
    # Load images and associated metadata
    self.data, self.metadata = workflow.load_ome_tiff(file=ome_tiff_file_path)
    
    # Subset data and metadata to the first 2 channels
    self.data = dict(list(self.data.items())[:2])
    
    self.metadata['channels'] = dict(list(self.metadata['channels'].items())[:2])
    
    # For each channel...
    for i, k in enumerate(self.data.keys()):
      
      # Extract config
      ch_config = self.config['channels'][i]
      
      # Perform robust normalization      
      self.data[k]['norm_img'] = workflow.robust_normalization(img=self.data[k]['img'], 
                                                               low_pct=ch_config['low_pct'], 
                                                               high_pct=ch_config['high_pct']) 
                    
    self.sig_output.emit(self.data, self.metadata)
          
# %% PredictWorker() ----

class PredictWorker(QObject):
  
  sig_message = Signal(str)                  
  sig_output = Signal(object)
  
  def __init__(self, 
               data: object, 
               ch_names: object,
               parent=None):
    
    super().__init__(parent)
    self.data = data
    self.ch_names = ch_names
    
  def run(self):
    
    # For each channel...    
    for k, v in self.ch_names.items():
      
      # Run StarDist
      self.data[v]['msk'] = models[k].predict_instances(img=normalize(self.data[v]['norm_img']),
                                                        prob_thresh=None,
                                                        nms_thresh=None)[0]
      
      # Compute submask area
      self.data[v]['msk_area'] = workflow.count_submsks_pixels(msk=self.data[v]['msk'])
              
    self.sig_output.emit(self.data)

# %% FilterSizeWorker() ----

class FilterSizeWorker(QObject):
  
  sig_message = Signal(str)                  
  sig_output = Signal(object)
  
  def __init__(self, 
               data: object, 
               config: object,
               ch_names: object,
               parent=None):
    
    super().__init__(parent)
    self.data = data
    self.config = config
    self.ch_names = ch_names
    
  def run(self):
    
    # For each channel...
    for k, v in self.ch_names.items():
      
      # Extract config
      ch_config = self.config['channels'][k]
      
      # Discard small submasks
      self.data[v]['filt_msk'] = workflow.discard_small_submsks(msk=self.data[v]['msk'], 
                                                                pix_dict=self.data[v]['msk_area'],
                                                                min_n_pix=ch_config['min_n_pix'])
      
      self.data[v]['filt_msk_edited'] = self.data[v]['filt_msk'].copy()
              
    self.sig_output.emit(self.data)
    
# %% ApplyEditsWorker() ----

class ApplyEditsWorker(QObject):
  
  sig_message = Signal(str)                  
  sig_output = Signal(object)
  
  def __init__(self, 
               layers: object,
               data: object, 
               config: object,
               ch_names: object,
               parent=None):
    
    super().__init__(parent)
    self.layers = layers
    self.data = data
    self.config = config
    self.ch_names = ch_names
    
  def run(self):
    
    # For each channel...
    for k in self.ch_names.values():
      
      # Retrieve user-edited cell predictions
      self.data[k]['filt_msk_edited'] = self.layers[k]['labels_layer'].data.copy()
      
      # Retrieve user-drawn cell shapes
      self.data[k]['user_submsks'] = self.layers[k]['shapes_layer'].data.copy()
      
      # If user drew cell shapes, add them to the edited cell predictions and compute their area
      if len(self.data[k]['user_submsks']) > 0:
                
        self.data[k]['filt_msk_edited'] = workflow.append_shapes_to_msk(msk=self.data[k]['filt_msk_edited'],
                                                                        shapes=self.data[k]['user_submsks'],
                                                                        start_idx=int(np.max(self.data[k]['filt_msk'])+1))
        
        tmp = self.data[k]['filt_msk_edited'].copy()
      
        tmp = np.where(tmp >= int(np.max(self.data[k]['filt_msk'])+1), tmp, 0)
      
        tmp = workflow.count_submsks_pixels(msk=tmp)
        
        for j in tmp.keys():
          
          if j not in self.data[k]['msk_area'].keys():
            
            self.data[k]['msk_area'][j] = tmp[j]
      
      self.data[k]['filt_cnt_edited'] = workflow.msk_to_cnts(msk=self.data[k]['filt_msk_edited'])
                    
    # Produce base merge image 
    ch0 = self.data[self.ch_names[0]]['norm_img'].astype(np.float32)
    ch1 = self.data[self.ch_names[1]]['norm_img'].astype(np.float32)

    c0_r, c0_g, c0_b, _ = to_rgba(self.config['channels'][0]['color'])
    c1_r, c1_g, c1_b, _ = to_rgba(self.config['channels'][1]['color'])

    scale = 0.8
    merge_r = scale * (ch0 * c0_r + ch1 * c1_r)
    merge_g = scale * (ch0 * c0_g + ch1 * c1_g)
    merge_b = scale * (ch0 * c0_b + ch1 * c1_b)

    merge_norm_img = np.stack([merge_r, merge_g, merge_b], axis=-1)
    merge_norm_img = np.clip(merge_norm_img, 0, 255).astype(np.uint8)
    
    self.data['merge'] = {'merge_norm_img': merge_norm_img.copy()}
    
    # Add ch0 / ch1 ROIs to merge image
    for i, j in zip([self.ch_names[0], 
                     self.ch_names[1]],
                    [(int(c0_r*255), int(c0_g*255), int(c0_b*255)), 
                     (int(c1_r*255), int(c1_g*255), int(c1_b*255))]):
      
      merge_norm_img = cv2.drawContours(image=merge_norm_img,
                                        contours=self.data[i]['filt_cnt_edited'], 
                                        contourIdx=-1,
                                        color=j,
                                        thickness=4)

    self.data['merge']['merge_norm_img_rois'] = merge_norm_img
    
    self.sig_output.emit(self.data)
        
# %% OverlapWorker() ----

class OverlapWorker(QObject):
  
  sig_message = Signal(str)                  
  sig_output = Signal(object)
  
  def __init__(self, 
               data: object, 
               config: object,
               ch_names: object,
               parent=None):
    
    super().__init__(parent)
    self.data = data
    self.config = config
    self.ch_names = ch_names
    self.elem_list = None
    
  def run(self):
    
    ch0_nm = self.ch_names[0]
    ch1_nm = self.ch_names[1]
    
    # Compute cell status 
    self.data[ch0_nm][f'{ch1_nm}_status'], self.data[ch1_nm][f'{ch0_nm}_status'] = workflow.get_submsks1_submsks2_status(
      msk1=self.data[ch0_nm]['filt_msk_edited'], 
      msk2=self.data[ch1_nm]['filt_msk_edited'], 
      min_pct_ovl_1by2=self.config['min_pct_ovl_ch0_by_ch1'],
      min_pct_ovl_2by1=self.config['min_pct_ovl_ch1_by_ch0'],
      submsks1_pix_dict=self.data[ch0_nm]['msk_area'],
      submsks2_pix_dict=self.data[ch1_nm]['msk_area']
      )
    
    # For each channel...
    for i, j in zip([ch0_nm, ch1_nm],
                    [ch1_nm, ch0_nm]):
      
      # Produce cell status summary
      summary = {k:len(v) for k, v in self.data[i][f'{j}_status'].items()}
      summary['total'] = sum(summary.values())
      self.data[i]['summary'] = summary
      
      # Extract submasks and contours
      self.data[i]['submsks'], self.data[i]['cnt']  = workflow.status_dict_to_submsks_and_cnts(
        msk=self.data[i]['filt_msk_edited'],
        status_dict=self.data[i][f'{j}_status']
        )
    
    # Add status ROIs to merge image
    c0_r, c0_g, c0_b, _ = to_rgba(self.config['channels'][0]['color'])
    c1_r, c1_g, c1_b, _ = to_rgba(self.config['channels'][1]['color'])
        
    status_colors_ch0 = {'neg': (int(c0_r*255), int(c0_g*255), int(c0_b*255)),
                         'pos': (0, 255, 0),
                         'amb': (255, 255, 0)}
    
    merge_norm_img = self.data['merge']['merge_norm_img'].copy()
    
    for i, j in status_colors_ch0.items():
      
      merge_norm_img = cv2.drawContours(image=merge_norm_img,
                                        contours=self.data[ch0_nm]['cnt'][i], 
                                        contourIdx=-1,
                                        color=j,
                                        thickness=4)

    merge_norm_img = cv2.drawContours(image=merge_norm_img,
                                      contours=self.data[ch1_nm]['cnt']['neg'],
                                      contourIdx=-1,
                                      color=(int(c1_r*255), int(c1_g*255), int(c1_b*255)),
                                      thickness=4)

    self.data['merge']['merge_norm_img_status'] = merge_norm_img
    
    self.sig_output.emit(self.data)
    
