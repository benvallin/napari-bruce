# %% Set up ----

# Load dependencies
import sys
import os
import copy
import numpy as np
import cv2
import json
import importlib.resources
from matplotlib.colors import to_rgba
from pathlib import Path
from qtpy.QtWidgets import QPushButton, QWidget, QLabel, QVBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox, QHBoxLayout, QApplication, QSizePolicy, QStyle, QComboBox
from qtpy.QtCore import Signal, QObject, QThread, QTimer, Qt
from enum import Enum
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
  
windows = sys.platform.startswith('win')

if windows:
  
  from tensorflow.config import list_physical_devices
  from tensorflow.config.experimental import set_memory_growth
    
  gpus = list_physical_devices('GPU')
    
  if gpus:  
    
    for gpu in gpus:
    
      try:
    
        set_memory_growth(gpu, True)
    
      except Exception:
    
        pass
                
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

# Signal readiness
ready_path = os.environ.get('NAPARI_BRUCE_READY_FILE')

if ready_path:
  
  Path(ready_path).touch(exist_ok=True)
        
# %% ParamValueBox() ----

class ParamValueBox(QWidget):
  
  valueChanged = Signal(float)
  
  def __init__(self, 
               label: str, 
               default: int | float = 0,
               min_val: int | float = 0, 
               max_val: int | float = 0, 
               decimals: int = 0, 
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

# %% ElementConfigBox() ----

class ElementConfigBox(QWidget):
  
  valueChanged = Signal(dict)  
  
  def __init__(self,
               label: str = '',
               n_collect: int = 0,
               min_n_collect: int = 0,
               max_n_collect: int = 1000,
               tube_id: str = '',
               tube_options: list[str] = configuration.list_tube_ids(),
               laser_function: str = '',
               laser_options: list[str] = configuration.list_laser_functions(),
               parent=None):
    
    super().__init__(parent)
    
    # general layout
    main_layout = QVBoxLayout(self)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(4)
    
    # label
    self.base_label = label
    self.label = QLabel(label)
    self.label.setStyleSheet('font-weight: bold')
    main_layout.addWidget(self.label)
    
    # n cells
    row1 = QHBoxLayout()
    row1.addWidget(QLabel('n collect'))
    
    self.spin_n = QSpinBox()
    self.spin_n.setRange(min_n_collect, max_n_collect)
    self.spin_n.setValue(n_collect)
    row1.addWidget(self.spin_n)
    
    main_layout.addLayout(row1)
    
    # tube ID
    row2 = QHBoxLayout()
    row2.addWidget(QLabel('tube ID'))
    
    self.combo_tube = QComboBox()
    self.combo_tube.addItems(tube_options)
    self.combo_tube.setCurrentText(tube_id)
    row2.addWidget(self.combo_tube)
    
    main_layout.addLayout(row2)
    
    # laser function
    row3 = QHBoxLayout()
    row3.addWidget(QLabel('laser function')) 
    
    self.combo_laser = QComboBox()
    self.combo_laser.addItems(laser_options)
    self.combo_laser.setCurrentText(laser_function)
    row3.addWidget(self.combo_laser)

    main_layout.addLayout(row3)

    # signals
    self.spin_n.valueChanged.connect(self.emit_value)
    self.combo_tube.currentTextChanged.connect(self.emit_value)
    self.combo_laser.currentTextChanged.connect(self.emit_value)
    
  # methods
  def value(self) -> dict:
    
    collect = True if self.spin_n.value() > 0 else False
    
    output = {'collect': collect,
              'laser_function': self.combo_laser.currentText(),
              'tube_id': self.combo_tube.currentText()}
    
    return output

  def emit_value(self):
    
    self.valueChanged.emit(self.value())

# %% MessageLevel() ----

class MessageLevel(Enum):
  
  NONE = None
  INFO = QStyle.SP_MessageBoxInformation
  BUSY = QStyle.SP_BrowserReload  
  WORK = QStyle.SP_ComputerIcon
  CHECK = QStyle.SP_ArrowRight
  SAVE = QStyle.SP_DialogSaveButton
  WARNING = QStyle.SP_MessageBoxWarning
  ERROR = QStyle.SP_MessageBoxCritical
    
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
              'box_min_area_ch0',
              'box_min_area_ch1',
              'btn_apply_edits',
              'btn_overlap',
              'btn_filter_overlap',
              'box_min_pct_ovl_ch0_by_ch1',
              'box_min_pct_ovl_ch1_by_ch0',
              'box_elem_ch0pos_ch1neg',
              'box_elem_ch0pos_ch1pos',
              'box_elem_ch0pos_ch1amb',
              'box_elem_ch1pos_ch0neg',
              'box_elem_ch1pos_ch0amb',
              'btn_save']:
      
      setattr(self, i, None)
      
    for i in ['_predict_requested', 
              '_filter_size_requested',
              '_filter_overlap_requested',
              '_channel_mismatch']:
      
      setattr(self, i, False)  
            
    self.btn_select = QPushButton('Select file')
    self.btn_select.clicked.connect(self.on_select_clicked)
    
    # Create default layout    
    self.msg_icon = QLabel()
    self.msg_icon.setFixedSize(16, 16)
    self.msg_icon.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    self.msg_icon.setVisible(False)

    self.msg_text = QLabel('')
    self.msg_text.setWordWrap(True)
    self.msg_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    self.msg_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    
    self.msg_container = QWidget()
    self.msg_layout = QHBoxLayout(self.msg_container)
    self.msg_layout.setContentsMargins(0, 0, 0, 0)
    self.msg_layout.setSpacing(6)
    self.msg_layout.addWidget(self.msg_icon, alignment=Qt.AlignTop)
    self.msg_layout.addWidget(self.msg_text, stretch=1)
    
    self.header_layout = QVBoxLayout()  
    self.header_layout.addWidget(self.btn_select)
    self.header_layout.addSpacing(30)    
    self.header_layout.addWidget(self.msg_container)
    
    self.layout = QVBoxLayout()
    self.layout.addLayout(self.header_layout)
    self.layout.addStretch(1)  
    self.setLayout(self.layout) 
    self.setMinimumWidth(300)
    self.setMaximumWidth(1000)   
                
  # Define methods    
  def set_message(self, text: str, level: MessageLevel = MessageLevel.NONE):
    
    if self._channel_mismatch:
      
      text = f'{text}\n\n\nWARNING: channel name mismatch between image and config.'

    self.msg_text.setText(text)
    
    if level.value is None:
      
      self.msg_icon.setVisible(False)
      
    else:
      
      icon = QApplication.style().standardIcon(level.value)
      self.msg_icon.setPixmap(icon.pixmap(16, 16))
      self.msg_icon.setVisible(True)
    
  def on_clear_clicked(self, *args):
    
    # Send Clear message
    self._channel_mismatch = False
    
    self.set_message('Resetting napari-bruce...', 
                     MessageLevel.BUSY)
        
    # Reset attributes to default values
    self.config = copy.deepcopy(config)
    
    for i in ['btn_clear',
              'btn_load',
              'btn_predict',
              'btn_filter_size',
              'box_min_area_ch0',
              'box_min_area_ch1',
              'btn_apply_edits',
              'btn_overlap',
              'btn_filter_overlap',
              'box_min_pct_ovl_ch0_by_ch1',
              'box_min_pct_ovl_ch1_by_ch0',
              'box_elem_ch0pos_ch1neg',
              'box_elem_ch0pos_ch1pos',
              'box_elem_ch0pos_ch1amb',
              'box_elem_ch1pos_ch0neg',
              'box_elem_ch1pos_ch0amb',
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
    
    # Delay viewer reset
    QApplication.processEvents()
    
    QTimer.singleShot(50, self.reset_viewer_to_defaults)
  
  def reset_viewer_to_defaults(self):
    
    # Clear all layers 
    self.viewer.layers.clear()
    
    # Recreate 'Select file' button 
    self.btn_select = QPushButton('Select file')
    self.btn_select.clicked.connect(self.on_select_clicked)
    self.layout.insertWidget(0, self.btn_select)
    
    # Send empty message
    self.set_message('', 
                     MessageLevel.NONE)
       
  def on_select_clicked(self):
    
    # Record user-selected file path
    self.path = QFileDialog.getOpenFileName(parent=self, 
                                            caption='Select .zvi file',
                                            directory=str(Path(self.config['in_dir_path']).expanduser()),
                                            filter='Images (*.zvi)')[0]
    
    # Abort if user cancelled selection
    if self.path == '':
      
      self.set_message('.zvi file selection cancelled', 
                       MessageLevel.WARNING)
      
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
    self.btn_clear.clicked.connect(lambda checked=False: self.on_clear_clicked())
    
    for i, j in zip([0, 1, 2],
                    [self.btn_load,
                     self.btn_predict, 
                     self.btn_clear]):
      
      self.layout.insertWidget(i, j)
    
    self.set_message('.zvi file selected', 
                     MessageLevel.INFO)
  
  def on_load_clicked(self):
    
    # Remove 'Load images' button from viewer
    self.layout.removeWidget(self.btn_load)
    self.btn_load.hide()
    self.btn_load.deleteLater()    
    self.btn_load = None
    
    self.set_message('Loading images...', 
                     MessageLevel.WORK)
    
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

    # Ensure UI result handling
    self._load_worker.sig_output.connect(self.on_load_finished)

    # Make thread stop when worker emits sig_output
    self._load_worker.sig_output.connect(self._load_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._load_worker_thread.finished.connect(self._load_worker.deleteLater)

    # Delete thread itself on finished 
    self._load_worker_thread.finished.connect(self._load_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def reset_attrs():
      
      for i in ['_load_worker_thread', '_load_worker']:
        
        setattr(self, i, None)
        
    self._load_worker_thread.finished.connect(reset_attrs)

    # Start worker thread
    self._load_worker_thread.start()

  def on_load_finished(self, error: bool, error_msg: str, data: dict, metadata: dict):
    
    # Stop if LoadWorker failed
    if error is True:
      
      msg_intro = 'LoadWorker failed with the following error:\n\n'
      
      print(f'{msg_intro}{error_msg}')
      
      self._channel_mismatch = False
      
      self.set_message(f'{msg_intro}{error_msg}', 
                       MessageLevel.ERROR)
      
      self.btn_clear.setEnabled(True)
      
      return
        
    # Warn if channel names are not consistent between metadata and config 
    ch0_nm_ok = metadata['channels'][0]['name'] == self.config['channels'][0]['name']
    ch1_nm_ok = metadata['channels'][1]['name'] == self.config['channels'][1]['name']
    
    if not (ch0_nm_ok and ch1_nm_ok):
            
      self._channel_mismatch = True
      self.config['channels'][0]['name'] = metadata['channels'][0]['name']
      self.config['channels'][1]['name'] = metadata['channels'][1]['name']
              
    self.set_message('Updating viewer...', 
                     MessageLevel.BUSY)
    
    # Update data / metadata / ch_names attributes with load worker output
    self.data = data
    self.metadata = metadata
    self.ch_names = {}
    for i in [0, 1]:
      self.ch_names[i] = self.metadata['channels'][i]['name']
    
    # Delay viewer update
    QApplication.processEvents()
    
    QTimer.singleShot(50, self.update_viewer_on_load_finished)
    
  def update_viewer_on_load_finished(self):
          
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
    self.box_min_area_ch0 = ParamValueBox(label=f'min area (\u00B5m\u00B2) {self.ch_names[0]}', 
                                          default=self.config['channels'][0]['min_area_um2'],
                                          min_val=0.0, 
                                          max_val=10000.0)    
    self.box_min_area_ch0.valueChanged.connect(lambda x: self.on_min_area_changed(0, x))
      
    self.box_min_area_ch1 = ParamValueBox(label=f'min area (\u00B5m\u00B2) {self.ch_names[1]}', 
                                          default=self.config['channels'][1]['min_area_um2'],
                                          min_val=0.0, 
                                          max_val=10000.0)
    self.box_min_area_ch1.valueChanged.connect(lambda x: self.on_min_area_changed(1, x))
      
    for i, j in zip([0, 1], 
                    [self.box_min_area_ch0, 
                     self.box_min_area_ch1]):
      
      self.layout.insertWidget(i, j)
    
    # Reset the view
    QTimer.singleShot(50, self.viewer.reset_view)
    
    # Stop here if 'Load images' button was clicked
    if not self._predict_requested:
      
      # Re-enable 'Predict cells' and 'Clear' buttons
      for i in [self.btn_predict, 
                self.btn_clear]:
        
        i.setEnabled(True)
      
      self.set_message("Check images in viewer\n\nAdjust min size thresholds (optional)\n\nPress 'Predict cells' when ready", 
                       MessageLevel.CHECK)
          
    # Alternatively, proceed with prediction if 'Predict cells' button was clicked
    else:
      
      # Disable 'min n pix' boxes
      for i in [self.box_min_area_ch0,  
                self.box_min_area_ch1]:
        
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
      
      self.set_message('Loading images...', 
                       MessageLevel.WORK)

      
      # Trigger load worker thread 
      self.start_load_thread()
      
      return
    
    # Alternatively, proceed with prediction if 'Load images' button was clicked previously
    self.on_img_loaded_and_predict_clicked()
  
  def on_img_loaded_and_predict_clicked(self):
    
    self.set_message('Predicting objects...', 
                     MessageLevel.WORK)
    
    # Trigger predict worker thread 
    self.start_predict_thread()
          
  def start_predict_thread(self):
    
    # Disable 'min n pix' boxes and 'Clear' button
    for i in [self.box_min_area_ch0, 
              self.box_min_area_ch1,
              self.btn_clear]:
      
      i.setEnabled(False)
            
    # Create thread and worker
    self._predict_worker_thread = QThread()
    self._predict_worker = PredictWorker(self.data, self.metadata, self.ch_names)
    self._predict_worker.moveToThread(self._predict_worker_thread)
    
    # Start worker when thread starts
    self._predict_worker_thread.started.connect(self._predict_worker.run)

    # Ensure UI result handling
    self._predict_worker.sig_output.connect(self.on_predict_finished)

    # Make thread stop when worker emits sig_output
    self._predict_worker.sig_output.connect(self._predict_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._predict_worker_thread.finished.connect(self._predict_worker.deleteLater)

    # Delete thread itself on finished 
    self._predict_worker_thread.finished.connect(self._predict_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def reset_attrs():
      
      for i in ['_predict_worker_thread', '_predict_worker']:
        
        setattr(self, i, None)
        
    self._predict_worker_thread.finished.connect(reset_attrs)

    # Start worker thread
    self._predict_worker_thread.start()
    
  def on_predict_finished(self, error: bool, error_msg: str, data: dict):
    
    # Stop if PredictWorker failed
    if error is True:
      
      msg_intro = 'PredictWorker failed with the following error:\n\n'
      
      print(f'{msg_intro}{error_msg}')
      
      self._channel_mismatch = False
            
      self.set_message(f'{msg_intro}{error_msg}', 
                       MessageLevel.ERROR)
      
      self.btn_clear.setEnabled(True)
      
      return
    
    # Update data attribute with predict worker output
    self.data = data
    
    self.set_message('Filtering out small objects...', 
                     MessageLevel.WORK)
    
    # Trigger filter size worker thread 
    self.start_filter_size_thread()    
    
  def on_filter_size_clicked(self):
    
    # Record that size filtering has been requested
    self._filter_size_requested = True
    
    self.set_message('Re-filtering out small objects...', 
                     MessageLevel.WORK)
    
    # Trigger filter size worker thread 
    self.start_filter_size_thread()
            
  def start_filter_size_thread(self):
    
    # Disable 'min n pix boxes', 'Adjust size filter', 'Apply edits' and 'Clear' buttons
    # => 'Adjust size filter' and 'Apply edits' buttons do not exist yet if predictions are returned for the first time 
    for i in [self.box_min_area_ch0, 
              self.box_min_area_ch1,
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

    # Ensure UI result handling
    self._filter_size_worker.sig_output.connect(self.on_filter_size_finished)

    # Make thread stop when worker emits sig_output
    self._filter_size_worker.sig_output.connect(self._filter_size_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._filter_size_worker_thread.finished.connect(self._filter_size_worker.deleteLater)

    # Delete thread itself on finished 
    self._filter_size_worker_thread.finished.connect(self._filter_size_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def reset_attrs():
      
      for i in ['_filter_size_worker_thread', '_filter_size_worker']:
        
        setattr(self, i, None)
        
    self._filter_size_worker_thread.finished.connect(reset_attrs)

    # Start worker thread
    self._filter_size_worker_thread.start()
    
  def on_filter_size_finished(self, error: bool, error_msg: str, data: dict):    
    
    # Stop if FilterSizeWorker failed
    if error is True:
      
      msg_intro = 'FilterSizeWorker failed with the following error:\n\n'
      
      print(f'{msg_intro}{error_msg}')
      
      self._channel_mismatch = False
            
      self.set_message(f'{msg_intro}{error_msg}', 
                       MessageLevel.ERROR)
      
      self.btn_clear.setEnabled(True)
      
      return
    
    # Update data attribute with filter size worker output
    self.data = data
    
    # Delay viewer update    
    self.set_message('Updating viewer...', 
                     MessageLevel.BUSY)
    
    QApplication.processEvents()
    
    QTimer.singleShot(50, self.update_viewer_on_filter_size_finished)
  
  def update_viewer_on_filter_size_finished(self):
    
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
    for i in [self.box_min_area_ch0, 
              self.box_min_area_ch1,
              self.btn_filter_size,
              self.btn_apply_edits,
              self.btn_clear]:
      
      i.setEnabled(True)
    
    self.set_message("Check images in viewer\n\nEdit cell contours (optional)\n\nPress 'Apply edits' when finished",
                     MessageLevel.CHECK)
        
  def on_apply_edits_clicked(self):
    
    # Remove 'min n pix' boxes, 'Adjust size filter' and 'Apply edits' buttons from viewer
    for i in ['box_min_area_ch0',
              'box_min_area_ch1',
              'btn_filter_size',
              'btn_apply_edits']:
      
      j = getattr(self, i, None)
      self.layout.removeWidget(j)
      j.hide()
      j.deleteLater()    
      setattr(self, i, None)
    
    self.set_message('Applying edits...', 
                     MessageLevel.WORK)
    
    # Trigger apply edits worker thread 
    self.start_apply_edits_thread()
      
  def start_apply_edits_thread(self):
    
    # Disable 'Clear' button
    self.btn_clear.setEnabled(False)
    
    # Create thread and worker
    self._apply_edits_worker_thread = QThread()
    self._apply_edits_worker = ApplyEditsWorker(self.layers, self.data, self.metadata, self.config, self.ch_names)
    self._apply_edits_worker.moveToThread(self._apply_edits_worker_thread)
    
    # Start worker when thread starts
    self._apply_edits_worker_thread.started.connect(self._apply_edits_worker.run)

    # Ensure UI result handling
    self._apply_edits_worker.sig_output.connect(self.on_apply_edits_finished)

    # Make thread stop when worker emits sig_output
    self._apply_edits_worker.sig_output.connect(self._apply_edits_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._apply_edits_worker_thread.finished.connect(self._apply_edits_worker.deleteLater)

    # Delete thread itself on finished 
    self._apply_edits_worker_thread.finished.connect(self._apply_edits_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def reset_attrs():
      
      for i in ['_apply_edits_worker_thread', '_apply_edits_worker']:
        
        setattr(self, i, None)
        
    self._apply_edits_worker_thread.finished.connect(reset_attrs)

    # Start worker thread
    self._apply_edits_worker_thread.start()
    
  def on_apply_edits_finished(self, error: bool, error_msg: str, data: dict):
    
    # Stop if ApplyEditsWorker failed
    if error is True:
      
      msg_intro = 'ApplyEditsWorker failed with the following error:\n\n'
      
      print(f'{msg_intro}{error_msg}')
      
      self._channel_mismatch = False
            
      self.set_message(f'{msg_intro}{error_msg}', 
                       MessageLevel.ERROR)
      
      self.btn_clear.setEnabled(True)
      
      return
    
    # Update data attribute with apply edits worker output
    self.data = data
    
    # Delay viewer update
    self.set_message('Updating viewer...', 
                     MessageLevel.BUSY)
    
    QApplication.processEvents()
    
    QTimer.singleShot(50, self.update_viewer_on_apply_edits_finished)
  
  def update_viewer_on_apply_edits_finished(self):
    
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
    self.box_min_pct_ovl_ch0_by_ch1 = ParamValueBox(label=f'min % overlap {self.ch_names[0]} / {self.ch_names[1]}', 
                                                    default=self.config['min_pct_ovl_ch0_by_ch1'],
                                                    min_val=0.0, 
                                                    max_val=100.0)
    self.box_min_pct_ovl_ch0_by_ch1.valueChanged.connect(lambda x: self.on_min_pct_ovl_changed('min_pct_ovl_ch0_by_ch1', x))
    
    self.box_min_pct_ovl_ch1_by_ch0 = ParamValueBox(label=f'min % overlap {self.ch_names[1]} / {self.ch_names[0]}', 
                                                    default=self.config['min_pct_ovl_ch1_by_ch0'],
                                                    min_val=0.0, 
                                                    max_val=100.0)
    self.box_min_pct_ovl_ch1_by_ch0.valueChanged.connect(lambda x: self.on_min_pct_ovl_changed('min_pct_ovl_ch1_by_ch0', x))

    self.btn_overlap = QPushButton('Find overlaps')
    self.btn_overlap.clicked.connect(self.on_overlap_clicked)
    
    for i, j in zip([0, 1, 2],
                    [self.box_min_pct_ovl_ch0_by_ch1,
                     self.box_min_pct_ovl_ch1_by_ch0, 
                     self.btn_overlap]):
      
      self.layout.insertWidget(i, j)
    
    # Re-enable 'Clear' button
    self.btn_clear.setEnabled(True)
    
    self.set_message("Adjust overlap thresholds (optional)\n\nPress 'Find overlaps' when ready", 
                     MessageLevel.CHECK)
       
  def on_overlap_clicked(self):
    
    # Remove 'Find overlaps' button from viewer
    self.layout.removeWidget(self.btn_overlap)
    self.btn_overlap.hide()
    self.btn_overlap.deleteLater()    
    self.btn_overlap = None
    
    self.set_message('Computing overlaps...', 
                     MessageLevel.WORK)
    
    # Trigger overlap worker thread   
    self.start_overlap_thread()
      
  def on_filter_overlap_clicked(self):
    
    # Record that % overlap filtering has been requested
    self._filter_overlap_requested = True
    
    self.set_message('Re-computing overlaps...', 
                     MessageLevel.WORK)
    
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

    # Ensure UI result handling
    self._overlap_worker.sig_output.connect(self.on_overlap_finished)

    # Make thread stop when worker emits sig_output
    self._overlap_worker.sig_output.connect(self._overlap_worker_thread.quit)

    # Ensure worker is deleted using thread's finished() signal
    self._overlap_worker_thread.finished.connect(self._overlap_worker.deleteLater)

    # Delete thread itself on finished 
    self._overlap_worker_thread.finished.connect(self._overlap_worker_thread.deleteLater)

    # Reset worker-related attributes to default values on finished 
    def reset_attrs():
      
      for i in ['_overlap_worker_thread', '_overlap_worker']:
        
        setattr(self, i, None)
        
    self._overlap_worker_thread.finished.connect(reset_attrs)

    # Start worker thread
    self._overlap_worker_thread.start()
    
  def on_overlap_finished(self, error: bool, error_msg: str, data: dict):
    
    # Stop if OverlapWorker failed
    if error is True:
      
      msg_intro = 'OverlapWorker failed with the following error:\n\n'
      
      print(f'{msg_intro}{error_msg}')
      
      self._channel_mismatch = False
            
      self.set_message(f'{msg_intro}{error_msg}', 
                       MessageLevel.ERROR)
      
      self.btn_clear.setEnabled(True)
      
      return
    
    # Update data attribute with overlap worker output
    self.data = data
    
    # Delay viewer update
    self.set_message('Updating viewer...', 
                     MessageLevel.BUSY)
    
    QApplication.processEvents()
    
    QTimer.singleShot(50, self.update_viewer_on_overlap_finished)
  
  def update_viewer_on_overlap_finished(self):
    
    # Update merge image layer
    self.layers['merge'].visible = False
    
    self.layers['merge'].data = self.data['merge']['merge_norm_img_status']
    
    self.layers['merge'].visible = True
    
    # Add 'Adjust overlap filter' button, 'element' boxes and 'Save results' button to viewer if overlaps are returned for the first time
    if not self._filter_overlap_requested:
      
      self.btn_filter_overlap = QPushButton('Adjust overlap filter')
      self.btn_filter_overlap.clicked.connect(self.on_filter_overlap_clicked)
      
      self.btn_save = QPushButton('Save results')
      self.btn_save.clicked.connect(self.on_save_clicked)
      
      plus = '\u207A'
      minus = '\u207B'  
      
      self.box_elem_ch0pos_ch1neg = ElementConfigBox(label=f'{self.ch_names[0]}{plus} / {self.ch_names[1]}{minus}')
      
      self.box_elem_ch0pos_ch1neg.valueChanged.connect(lambda x: self.on_elem_params_changed('ch0-pos/ch1-neg', x))
      
      self.box_elem_ch0pos_ch1pos = ElementConfigBox(label=f'{self.ch_names[0]}{plus} / {self.ch_names[1]}{plus}')
      
      self.box_elem_ch0pos_ch1pos.valueChanged.connect(lambda x: self.on_elem_params_changed('ch0-pos/ch1-pos', x))
      
      self.box_elem_ch0pos_ch1amb = ElementConfigBox(label=f'{self.ch_names[0]}{plus} / {self.ch_names[1]}-amb')
      
      self.box_elem_ch0pos_ch1amb.valueChanged.connect(lambda x: self.on_elem_params_changed('ch0-pos/ch1-amb', x))
      
      self.box_elem_ch1pos_ch0neg = ElementConfigBox(label=f'{self.ch_names[1]}{plus} / {self.ch_names[0]}{minus}')
      
      self.box_elem_ch1pos_ch0neg.valueChanged.connect(lambda x: self.on_elem_params_changed('ch1-pos/ch0-neg', x))
      
      self.box_elem_ch1pos_ch0amb = ElementConfigBox(label=f'{self.ch_names[1]}{plus} / {self.ch_names[0]}-amb')
      
      self.box_elem_ch1pos_ch0amb.valueChanged.connect(lambda x: self.on_elem_params_changed('ch1-pos/ch0-amb', x))
        
      for i, j in zip([2, 3, 4, 5, 6, 7, 8],
                      [self.btn_filter_overlap,
                       self.box_elem_ch0pos_ch1neg,
                       self.box_elem_ch0pos_ch1pos,
                       self.box_elem_ch0pos_ch1amb,
                       self.box_elem_ch1pos_ch0neg,
                       self.box_elem_ch1pos_ch0amb,
                       self.btn_save]):
        
        self.layout.insertWidget(i, j)
        
    # Re-enable 'min % ovl' and 'element' boxes, 'Adjust overlap filter', 'Save results' and 'Clear' buttons
    # => 'Adjust overlap filter' button, 'element' boxes and 'Save results' button do not exist yet if overlaps are returned for the first time 
    for i in [self.box_min_pct_ovl_ch0_by_ch1,
              self.box_min_pct_ovl_ch1_by_ch0, 
              self.btn_filter_overlap,
              self.box_elem_ch0pos_ch1neg,
              self.box_elem_ch0pos_ch1pos,
              self.box_elem_ch0pos_ch1amb,
              self.box_elem_ch1pos_ch0neg,
              self.box_elem_ch1pos_ch0amb,
              self.btn_save,
              self.btn_clear]:
      
      i.setEnabled(True)
    
    # Update 'element' boxes 
    self.update_elem_boxes()
    
    self.set_message("Adjust element list parameters (optional)\n\nPress 'Save results' when finished", 
                     MessageLevel.CHECK)
  
  def on_save_clicked(self):
    
    # Read content of 'element' boxes
    self.read_elem_boxes()
    
    # Define output directory path 
    out_dir_path = os.path.join(Path(self.config['out_dir_path']).expanduser(), 
                                self.metadata['img_nm'])
    
    # Construct element list and write to file 
    self.elem_list = workflow.make_elem_txt(data_dict=self.data, 
                                            metadata_dict=self.metadata,
                                            config_dict=self.config)
    
    with open(os.path.join(out_dir_path, 'elem_list.txt'), 'w') as f:
      f.write(self.elem_list)
      
    # Write data and metadata to file
    workflow.pickle_data(data=self.data, filename=os.path.join(out_dir_path, 'data.pkl'))
    
    workflow.pickle_data(data=self.metadata, filename=os.path.join(out_dir_path, 'metadata.pkl'))
    
    # Write config to file
    with open(os.path.join(out_dir_path, 'config.json'), 'w') as f:
      json.dump(self.config, f, indent=2)
    
    self.set_message(f"Results saved at:\n{self.config['out_dir_path']}\n\nIn subfolder:\n{self.metadata['img_nm']}",
                     MessageLevel.SAVE)
            
  def on_min_area_changed(self, key: int, value: float):

    self.config['channels'][key]['min_area_um2'] = float(value)
    
  def on_min_pct_ovl_changed(self, key: str, value: float):

    self.config[key] = float(value)
    
  def on_elem_params_changed(self, key: str, values: dict):
    
    self.config['elements'][key].update(values)
  
  def update_elem_boxes(self):
    
    primary_channel_id = [0, 0, 0, 1, 1]
    
    secondary_channel_status = ['neg', 'pos', 'amb', 'neg', 'amb']
    
    population = ['ch0-pos/ch1-neg', 
                  'ch0-pos/ch1-pos', 
                  'ch0-pos/ch1-amb', 
                  'ch1-pos/ch0-neg', 
                  'ch1-pos/ch0-amb']

    box = [self.box_elem_ch0pos_ch1neg,
           self.box_elem_ch0pos_ch1pos,
           self.box_elem_ch0pos_ch1amb,
           self.box_elem_ch1pos_ch0neg,
           self.box_elem_ch1pos_ch0amb]
    
    for i, j, k, l in zip(primary_channel_id, secondary_channel_status, population, box):
      
      max_n_collect = self.data[self.ch_names[i]]['summary'][j] 
      n_collect = max_n_collect if self.config['elements'][k]['collect'] is True else 0
      laser_function = self.config['elements'][k]['laser_function'] if max_n_collect > 0 else ''
      tube_id = self.config['elements'][k]['tube_id'] if max_n_collect > 0 else ''
      color = self.config['elements'][k]['color']
      
      if l is not None:
        
        base_label = l.base_label
        label = l.label
        label.setText(f'{base_label} ({max_n_collect})')
        label.setStyleSheet(f'font-weight: bold; color: {color}')
        
        spin_n = l.spin_n
        spin_n.blockSignals(True)
        spin_n.setMaximum(max_n_collect)
        spin_n.setValue(n_collect)
        spin_n.blockSignals(False)
        
        combo_laser = l.combo_laser
        combo_laser.blockSignals(True)
        combo_laser.setCurrentText(laser_function)
        combo_laser.blockSignals(False)
        
        combo_tube = l.combo_tube
        combo_tube.blockSignals(True)
        combo_tube.setCurrentText(tube_id)
        combo_tube.blockSignals(False)
  
  def read_elem_boxes(self):
    
    primary_channel_id = [0, 0, 0, 1, 1]
    
    secondary_channel_status = ['neg', 'pos', 'amb', 'neg', 'amb']
    
    box = [self.box_elem_ch0pos_ch1neg,
           self.box_elem_ch0pos_ch1pos,
           self.box_elem_ch0pos_ch1amb,
           self.box_elem_ch1pos_ch0neg,
           self.box_elem_ch1pos_ch0amb]
    
    for i, j, k in zip(primary_channel_id, secondary_channel_status, box):
      
      n_collect = k.spin_n.value()
      laser_function = k.combo_laser.currentText()
      tube_id = k.combo_tube.currentText()
      
      self.data[self.ch_names[i]]['summary'][f'{j}_collect'] = n_collect
      
      self.data[self.ch_names[i]]['summary'][f'{j}_laser_function'] = laser_function
      
      self.data[self.ch_names[i]]['summary'][f'{j}_tube_id'] = tube_id
    
    self.data[self.ch_names[0]]['summary'] = {k:self.data[self.ch_names[0]]['summary'][k]
                                              for k in ['total', 'neg', 'pos', 'amb', 
                                                        'neg_collect', 'pos_collect', 'amb_collect',
                                                        'neg_laser_function', 'pos_laser_function', 'amb_laser_function',
                                                        'neg_tube_id', 'pos_tube_id', 'amb_tube_id']}
    
    self.data[self.ch_names[1]]['summary'] = {k:self.data[self.ch_names[1]]['summary'][k]
                                              for k in ['total', 'neg', 'pos', 'amb', 
                                                        'neg_collect', 'amb_collect',
                                                        'neg_laser_function', 'amb_laser_function',
                                                        'neg_tube_id', 'amb_tube_id']}
        
# %% LoadWorker() ----

class LoadWorker(QObject):
                 
  sig_output = Signal(bool, str, object, object)
  
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
    
    try:
      
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
      
      error = False
      error_msg = ''    
                
    except Exception as e:
      
      error = True
      error_msg = f'{type(e).__name__}: {e}'
      
    self.sig_output.emit(error, error_msg, self.data, self.metadata)
          
# %% PredictWorker() ----

class PredictWorker(QObject):
  
  sig_output = Signal(bool, str, object)
  
  def __init__(self, 
               data: object, 
               metadata: object,
               ch_names: object,
               parent=None):
    
    super().__init__(parent)
    self.data = data
    self.metadata = metadata
    self.ch_names = ch_names
    
  def run(self):
    
    try:
      
      # For each channel...    
      for k, v in self.ch_names.items():
      
        # Run StarDist        
        img = normalize(self.data[v]['img'], 1, 99.8, axis=(0,1)) # Adjust this depending on training parameters
        
        n_tiles = workflow.choose_stardist_n_tiles(img)
      
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
        
          self.data[v]['msk'] = models[k].predict_instances(img=img,
                                                            prob_thresh=None,
                                                            nms_thresh=None, 
                                                            n_tiles=n_tiles)[0]
      
        # Compute submask area
        self.data[v]['msk_area'] = workflow.count_submsks_pixels(msk=self.data[v]['msk'])
        
        # Convert area in pixel^2 to µm^2
        px_area_um2 = self.metadata['image']['px_area_um2']
    
        self.data[v]['msk_area_um2'] = {k:v*px_area_um2 for k, v in self.data[v]['msk_area'].items()}
        
      error = False
      error_msg = ''    
                
    except Exception as e:
      
      error = True
      error_msg = f'{type(e).__name__}: {e}'
      
    self.sig_output.emit(error, error_msg, self.data)      

# %% FilterSizeWorker() ----

class FilterSizeWorker(QObject):
  
  sig_output = Signal(bool, str, object)
  
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
    
    try:
      
      # For each channel...
      for k, v in self.ch_names.items():
      
        # Extract config
        ch_config = self.config['channels'][k]
      
        # Discard small submasks
        self.data[v]['filt_msk'] = workflow.discard_small_submsks(msk=self.data[v]['msk'], 
                                                                  pix_dict=self.data[v]['msk_area_um2'],
                                                                  min_n_pix=ch_config['min_area_um2'])
      
        self.data[v]['filt_msk_edited'] = self.data[v]['filt_msk'].copy()
      
      error = False
      error_msg = ''    
                
    except Exception as e:
      
      error = True
      error_msg = f'{type(e).__name__}: {e}'        
    
    self.sig_output.emit(error, error_msg, self.data)    
    
# %% ApplyEditsWorker() ----

class ApplyEditsWorker(QObject):
  
  sig_output = Signal(bool, str, object)
  
  def __init__(self, 
               layers: object,
               data: object, 
               metadata: object,
               config: object,
               ch_names: object,
               parent=None):
    
    super().__init__(parent)
    self.layers = layers
    self.data = data
    self.metadata = metadata
    self.config = config
    self.ch_names = ch_names
    
  def run(self):
    
    try:
      
      px_area_um2 = self.metadata['image']['px_area_um2']
      
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
              
              self.data[k]['msk_area_um2'][j] = tmp[j]*px_area_um2 
      
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
                                          contours=[x for x in self.data[i]['filt_cnt_edited'].values()], 
                                          contourIdx=-1,
                                          color=j,
                                          thickness=4)

      self.data['merge']['merge_norm_img_rois'] = merge_norm_img

      error = False
      error_msg = ''    
                
    except Exception as e:
      
      error = True
      error_msg = f'{type(e).__name__}: {e}'        
    
    self.sig_output.emit(error, error_msg, self.data)
        
# %% OverlapWorker() ----

class OverlapWorker(QObject):
  
  sig_output = Signal(bool, str, object)
  
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
    
    try:
      
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
        self.data[i]['submsks'], self.data[i]['cnt'] = workflow.status_dict_to_submsks_and_cnts(
          msk=self.data[i]['filt_msk_edited'],
          status_dict=self.data[i][f'{j}_status'],
          cnt_dict=self.data[i]['filt_cnt_edited']
          )
    
      # Add status ROIs to merge image      
      c0pc1n_r, c0pc1n_g, c0pc1n_b, _ = tuple(int(i*255) for i in to_rgba(self.config['elements']['ch0-pos/ch1-neg']['color']))
      c0pc1p_r, c0pc1p_g, c0pc1p_b, _ = tuple(int(i*255) for i in to_rgba(self.config['elements']['ch0-pos/ch1-pos']['color']))
      c0pc1a_r, c0pc1a_g, c0pc1a_b, _ = tuple(int(i*255) for i in to_rgba(self.config['elements']['ch0-pos/ch1-amb']['color']))
      c1pc0n_r, c1pc0n_g, c1pc0n_b, _ = tuple(int(i*255) for i in to_rgba(self.config['elements']['ch1-pos/ch0-neg']['color']))
      c1pc0a_r, c1pc0a_g, c1pc0a_b, _ = tuple(int(i*255) for i in to_rgba(self.config['elements']['ch1-pos/ch0-amb']['color']))
      
      status_colors_ch0 = {'neg': (c0pc1n_r, c0pc1n_g, c0pc1n_b),
                           'pos': (c0pc1p_r, c0pc1p_g, c0pc1p_b),
                           'amb': (c0pc1a_r, c0pc1a_g, c0pc1a_b)}
      
      status_colors_ch1 = {'neg': (c1pc0n_r, c1pc0n_g, c1pc0n_b),
                           'amb': (c1pc0a_r, c1pc0a_g, c1pc0a_b)}
    
      merge_norm_img = self.data['merge']['merge_norm_img'].copy()
    
      for i, j in status_colors_ch0.items():
      
        merge_norm_img = cv2.drawContours(image=merge_norm_img,
                                          contours=[x for x in self.data[ch0_nm]['cnt'][i].values()], 
                                          contourIdx=-1,
                                          color=j,
                                          thickness=4)
        
      for i, j in status_colors_ch1.items():
        
        merge_norm_img = cv2.drawContours(image=merge_norm_img,
                                          contours=[x for x in self.data[ch1_nm]['cnt'][i].values()], 
                                          contourIdx=-1,
                                          color=j,
                                          thickness=4)

      self.data['merge']['merge_norm_img_status'] = merge_norm_img
      
      error = False
      error_msg = ''    
                
    except Exception as e:
      
      error = True
      error_msg = f'{type(e).__name__}: {e}'        
    
    self.sig_output.emit(error, error_msg, self.data)    
