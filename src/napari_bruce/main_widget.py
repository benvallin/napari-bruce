# %% Set up ----

# Load dependencies
import sys
import os
import copy
import json
import importlib.resources
import traceback
import numpy as np
from pathlib import Path
from enum import Enum, auto
from matplotlib.colors import to_rgba
import threading
import contextlib
from qtpy.QtWidgets import (
    QPushButton,
    QWidget,
    QLabel,
    QVBoxLayout,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QApplication,
    QSizePolicy,
    QStyle,
    QComboBox,
    QDialog,
    QScrollArea,
    QFrame,
    QCheckBox,
    QDialogButtonBox,
    QMessageBox,
)
from qtpy.QtCore import Signal, QObject, QThread, Qt
from qtpy.QtGui import QColor
from typing import TYPE_CHECKING
from . import configuration
from . import workflow
from .workflow import Population, POPULATIONS, POPULATION_BY_KEY

if TYPE_CHECKING:
    import napari

# Check Java
workflow.require_java()

# Load configuration
config = configuration.get_config()

# Signal program started
ready_path = os.environ.get("NAPARI_BRUCE_READY_FILE")
if ready_path:
    Path(ready_path).touch(exist_ok=True)

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


# %% WorkflowState() ----


class WorkflowState(Enum):

    SELECT_MODEL = auto()
    SELECT_FILE = auto()
    LOAD_IMAGE = auto()
    LOADING_IMAGE = auto()
    LOADING_IMAGE_FOR_PREDICT_ROI = auto()
    PREDICT_ROI = auto()
    IMAGE_LOADED_FOR_PREDICT_ROI = auto()
    PREDICTING_ROI = auto()
    APPLY_EDITS = auto()
    APPLYING_EDITS = auto()
    OVERLAPPING_ROI = auto()
    OVERLAP_FILTER_OR_SAVE = auto()
    UPDATE_OVERLAP_FILTER_OR_SAVE = auto()
    CLEARED = auto()


# %% ViewerState() ----


class ViewerState(Enum):

    NO_IMAGE = auto()
    IMAGE_LOADED = auto()
    ROI_EDITING = auto()
    LOCKED = auto()


# %% ControlState() ----


class ControlState:

    workflow_state: WorkflowState | None
    viewer_state: ViewerState | None

    def __init__(self):
        self.reset()

    def reset(self):
        self.workflow_state = None
        self.viewer_state = None


# %% WorkflowData() ----


class WorkflowData:

    path: str
    data: dict
    metadata: dict
    ch_names: dict[int, str]
    channel_mismatch: bool

    def __init__(self):
        self.reset()

    def reset(self):
        self.path = ""
        self.data = {}
        self.metadata = {}
        self.ch_names = {}
        self.channel_mismatch = False


# %% UIComponents() ----


class UIComponents:

    box_model_ch0: QWidget | None
    box_model_ch1: QWidget | None
    combo_model_ch0: QComboBox | None
    combo_model_ch1: QComboBox | None
    btn_load_models: QPushButton | None
    btn_change_models: QPushButton | None
    btn_select: QPushButton | None
    btn_clear: QPushButton | None
    btn_load: QPushButton | None
    btn_predict: QPushButton | None
    btn_filter_size: QPushButton | None
    btn_discard_additions: QPushButton | None
    btn_discard_deletions: QPushButton | None
    btn_apply_edits: QPushButton | None
    btn_repredict: QPushButton | None
    btn_overlap_filter: QPushButton | None
    btn_edit_rois: QPushButton | None
    btn_save: QPushButton | None
    box_low_pct_ch0: "ParamValueBox | None"
    box_high_pct_ch0: "ParamValueBox | None"
    box_low_pct_ch1: "ParamValueBox | None"
    box_high_pct_ch1: "ParamValueBox | None"
    box_min_area_ch0: "ParamValueBox | None"
    box_min_area_ch1: "ParamValueBox | None"
    box_min_pct_ovl_ch0_by_ch1: "ParamValueBox | None"
    box_min_pct_ovl_ch1_by_ch0: "ParamValueBox | None"
    box_elems: "dict[str, ElementConfigBox]"
    btns_choose_rois: dict[str, QPushButton]

    def __init__(self):
        self.reset()

    def reset(self):
        self.box_model_ch0 = None
        self.box_model_ch1 = None
        self.combo_model_ch0 = None
        self.combo_model_ch1 = None
        self.btn_load_models = None
        self.btn_change_models = None
        self.btn_select = None
        self.btn_clear = None
        self.btn_load = None
        self.btn_predict = None
        self.btn_filter_size = None
        self.btn_discard_additions = None
        self.btn_discard_deletions = None
        self.btn_apply_edits = None
        self.btn_repredict = None
        self.btn_overlap_filter = None
        self.btn_edit_rois = None
        self.btn_save = None
        self.box_low_pct_ch0 = None
        self.box_high_pct_ch0 = None
        self.box_low_pct_ch1 = None
        self.box_high_pct_ch1 = None
        self.box_min_area_ch0 = None
        self.box_min_area_ch1 = None
        self.box_min_pct_ovl_ch0_by_ch1 = None
        self.box_min_pct_ovl_ch1_by_ch0 = None
        self.box_elems = {}
        self.btns_choose_rois = {}


# %% ParamValueBox() ----


class ParamValueBox(QWidget):

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        default: int | float = 0,
        min_val: int | float = 0,
        max_val: int | float = 0,
        decimals: int = 0,
        parent=None,
    ):

        super().__init__(parent)

        # General layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label and spin box
        self._label = QLabel(label)
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setRange(min_val, max_val)
        self._spin.setValue(default)
        layout.addWidget(self._label)
        layout.addWidget(self._spin)
        self.setFixedHeight(QPushButton().sizeHint().height())

        # Signal
        self._spin.valueChanged.connect(self.valueChanged.emit)


# %% ElementConfigBox() ----


class ElementConfigBox(QWidget):

    valueChanged = Signal(dict)
    nCollectChanged = Signal(int)

    def __init__(
        self,
        label: str = "",
        n_collect: int = 0,
        min_n_collect: int = 0,
        max_n_collect: int = 1000,
        tube_id: str = "",
        tube_options: list[str] = configuration.list_tube_ids(),
        laser_function: str = "",
        laser_options: list[str] = configuration.list_laser_functions(),
        parent=None,
    ):

        super().__init__(parent)

        # General layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Population label
        self.base_label = label
        self.label = QLabel(label)
        self.label.setStyleSheet("font-weight: bold")
        main_layout.addWidget(self.label)

        # ROI selection - n collect
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("n collect"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(min_n_collect, max_n_collect)
        self.spin_n.setValue(n_collect)
        self.spin_n.setMaximumWidth(60)
        row1.addWidget(self.spin_n)

        # ROI selection - Choose ROIs
        self.btn_choose_rois = QPushButton("Choose ROIs")
        row1.addWidget(self.btn_choose_rois, stretch=1)
        main_layout.addLayout(row1)

        # Tube ID
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("tube ID"))
        self.combo_tube = QComboBox()
        self.combo_tube.addItems(tube_options)
        self.combo_tube.setCurrentText(tube_id)
        row2.addWidget(self.combo_tube)
        main_layout.addLayout(row2)

        # Laser function
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("laser function"))
        self.combo_laser = QComboBox()
        self.combo_laser.addItems(laser_options)
        self.combo_laser.setCurrentText(laser_function)
        row3.addWidget(self.combo_laser)
        main_layout.addLayout(row3)

        # Signals
        self.spin_n.valueChanged.connect(self.nCollectChanged)
        self.spin_n.valueChanged.connect(self._emit_value)
        self.combo_tube.currentTextChanged.connect(self._emit_value)
        self.combo_laser.currentTextChanged.connect(self._emit_value)

    def value(self) -> dict:
        collect = True if self.spin_n.value() > 0 else False
        output = {
            "collect": collect,
            "laser_function": self.combo_laser.currentText(),
            "tube_id": self.combo_tube.currentText(),
        }
        return output

    def _emit_value(self):
        self.valueChanged.emit(self.value())


# %% ChooseROIsWindow() ----


class ChooseROIsWindow(QDialog):

    def __init__(
        self,
        title,
        rois,
        initial_selected,
        px_area_um2=1.0,
        ovl_key=None,
        ovl_label="",
        parent=None,
    ):

        super().__init__(parent)

        # General layout
        self.setWindowTitle(title)
        self.setMinimumSize(340, 460)
        layout = QVBoxLayout(self)

        # Select all / none buttons
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_none = QPushButton("Select none")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        # ROI checkboxes in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll_layout = QVBoxLayout(container)
        scroll_layout.setSpacing(2)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        self._checkboxes = {}
        self._areas_um2 = {}

        for rank, (roi_id, roi_data) in enumerate(rois):
            area_um2 = roi_data.get("area", 0) * px_area_um2
            self._areas_um2[roi_id] = area_um2
            label = f"#{rank + 1}  |  {area_um2:.0f} µm²"
            if ovl_key is not None and "summary" in roi_data:
                ovl = roi_data["summary"].get(ovl_key, 0)
                label += f"  |  {ovl_label} = {ovl:.1f}"
            roi_box = QCheckBox(label)
            roi_box.setChecked(roi_id in initial_selected)
            roi_box.stateChanged.connect(self._update_total_area)
            self._checkboxes[roi_id] = roi_box
            scroll_layout.addWidget(roi_box)

        scroll_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Summary message (n selected + total area)
        self._total_area_label = QLabel()
        self._total_area_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._total_area_label)
        self._update_total_area()

        # OK / cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_total_area(self, *args):
        total = sum(
            self._areas_um2[roi_id]
            for roi_id, roi_box in self._checkboxes.items()
            if roi_box.isChecked()
        )
        n = sum(1 for roi_box in self._checkboxes.values() if roi_box.isChecked())
        self._total_area_label.setText(f"Selected: {n}  |  Total area: {total:.0f} µm²")

    def _select_all(self):
        for roi_box in self._checkboxes.values():
            roi_box.setChecked(True)

    def _select_none(self):
        for roi_box in self._checkboxes.values():
            roi_box.setChecked(False)

    def selected_ids(self):
        return {
            roi_id
            for roi_id, roi_box in self._checkboxes.items()
            if roi_box.isChecked()
        }


# %% ChannelSelectWindow() ----


class ChannelSelectWindow(QDialog):
    """Pick which channel(s) an edit-discard action applies to (one or both)."""

    def __init__(self, title, channel_names, prompt="", parent=None):

        super().__init__(parent)

        self.setWindowTitle(title)
        layout = QVBoxLayout(self)

        if prompt:
            layout.addWidget(QLabel(prompt))

        # One checkbox per channel, both ticked by default (the common case).
        self._checkboxes = {}
        for idx in sorted(channel_names):
            cb = QCheckBox(channel_names[idx])
            cb.setChecked(True)
            self._checkboxes[idx] = cb
            layout.addWidget(cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_channels(self):
        return [idx for idx, cb in self._checkboxes.items() if cb.isChecked()]


# %% BaseWorker() ----


class BaseWorker(QObject):

    sig_success = Signal(dict)
    sig_error = Signal(dict)

    def run(self):
        try:
            output = self.compute()
            self.sig_success.emit(output)
        except Exception as e:
            error_payload = {
                "worker": type(self).__name__,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback": traceback.format_exc(),
            }
            print(error_payload)
            self.sig_error.emit(error_payload)

    def compute(self):
        raise NotImplementedError


# %% NormalizationWorker() ----


class NormalizationWorker(BaseWorker):

    def __init__(self, data: dict, ch_names: dict, config: dict, ch_indices: list, parent=None):

        super().__init__(parent)
        self.data = data
        self.ch_names = ch_names
        self.config = config
        self.ch_indices = ch_indices

    def compute(self):

        output = {}
        for i in self.ch_indices:
            ch_nm = self.ch_names[i]
            norm_img = workflow.robust_normalization(
                img=self.data[ch_nm]["img"],
                low_pct=self.config["channels"][i]["low_pct"],
                high_pct=self.config["channels"][i]["high_pct"],
            )
            output[i] = norm_img
        return output


# %% LoadModelWorker() ----


class _ThreadSuppressor:
    """Thread-locally suppressible proxy for sys.stdout / sys.stderr.

    Installed once (before the first model load) so that LoadModelWorker can
    silence StarDist / TF output in its own thread without touching other
    threads. Each thread has an independent 'silent' flag via threading.local,
    so silencing the preload thread never affects main-thread output.

    All stream attributes not handled explicitly are forwarded to the wrapped
    stream via __getattr__ so that the proxy is transparent to the rest of
    the application.
    """

    def __init__(self, stream):
        self._stream = stream
        self._local = threading.local()

    @contextlib.contextmanager
    def silence(self):
        self._local.silent = True
        try:
            yield
        finally:
            self._local.silent = False

    def write(self, s: str) -> int:
        if getattr(self._local, "silent", False):
            return len(s)
        return self._stream.write(s)

    def writelines(self, lines) -> None:
        if not getattr(self._local, "silent", False):
            self._stream.writelines(lines)

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        pass  # never close the underlying stream

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@contextlib.contextmanager
def _silence_model_output():
    """Silence stdout/stderr in the current thread during StarDist/TF model loading.

    Works only when sys.stdout/stderr are _ThreadSuppressor instances (installed
    by PluginManager.__init__). Silently no-ops otherwise.
    """
    targets = [s for s in (sys.stdout, sys.stderr) if isinstance(s, _ThreadSuppressor)]
    with contextlib.ExitStack() as stack:
        for t in targets:
            stack.enter_context(t.silence())
        yield


class LoadModelWorker(BaseWorker):

    def __init__(self, config: dict, parent=None):

        super().__init__(parent)
        self.config = config

    def _load_models(self) -> dict:
        from stardist.models import StarDist2D

        pretrained = [
            k
            for k, v in configuration.list_stardist_models().items()
            if v == "pretrained"
        ]
        models = {}
        for i in [0, 1]:
            model_nm = self.config["channels"][i]["stardist_model"]
            if model_nm in pretrained:
                models[i] = StarDist2D.from_pretrained(model_nm)
            else:
                stardist_models_dir_path = Path(
                    str(importlib.resources.files("napari_bruce")),
                    "stardist_models",
                )
                models[i] = StarDist2D(
                    None, name=model_nm, basedir=stardist_models_dir_path
                )
        return models

    def compute(self):

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # Set memory growth if windows + GPU
        windows = sys.platform.startswith("win")
        if windows:
            from tensorflow.config import list_physical_devices
            from tensorflow.config.experimental import set_memory_growth

            gpus = list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    try:
                        set_memory_growth(gpu, True)
                    except Exception:
                        pass

        with _silence_model_output():
            models = self._load_models()

        return models


# %% LoadWorker() ----


class LoadWorker(BaseWorker):

    def __init__(self, path: str, config: dict, parent=None):

        super().__init__(parent)
        self.path = path
        self.config = config

    def compute(self):

        # Define output paths
        out_dir_path = Path(
            Path(self.config["out_dir_path"]).expanduser(), Path(self.path).stem
        )
        ome_tiff_file_path = Path(out_dir_path, Path(self.path).stem + ".ome.tiff")

        # Convert PALM .zvi file to OME-TIFF
        workflow.convert_zvi_to_ome(
            file=self.path,
            out_dir_path=out_dir_path,
            jar_pkg="napari_bruce.bioformats",
            jar_name="bioformats_package.jar",
        )

        # Load images and associated metadata
        raw_data, raw_metadata = workflow.load_ome_tiff(file=ome_tiff_file_path)

        # Subset data and metadata to the first 2 channels
        data = dict(list(raw_data.items())[:2])
        metadata = {
            **raw_metadata,
            "channels": dict(list(raw_metadata["channels"].items())[:2]),
        }

        # For each channel...
        for i, k in enumerate(data.keys()):
            # Extract config
            ch_config = self.config["channels"][i]
            # Perform robust normalization
            norm_img = workflow.robust_normalization(
                img=data[k]["img"],
                low_pct=ch_config["low_pct"],
                high_pct=ch_config["high_pct"],
            )
            data[k] = {**data[k], "norm_img": norm_img}

        return {"data": data, "metadata": metadata}


# %% PredictFilterWorker() ----


class PredictFilterWorker(BaseWorker):

    def __init__(
        self,
        data: dict,
        metadata: dict,
        config: dict,
        ch_names: dict,
        models: dict,
        do_predict: bool | None = None,
        parent=None,
    ):

        super().__init__(parent)
        self.data = data
        self.metadata = metadata
        self.config = config
        self.ch_names = ch_names
        self.models = models
        self.do_predict = do_predict

    def compute(self):

        output = {}
        filt_msk_changed = {}
        px_area_um2 = self.metadata["image"]["px_area_um2"]

        for k, v in self.ch_names.items():

            # Perform prediction if do_predict
            if self.do_predict:
                # Run StarDist
                img = self.data[v]["norm_img"].astype(np.float32) / 255.0
                n_tiles = workflow.choose_stardist_n_tiles(img)
                with _silence_model_output():
                    msk = self.models[k].predict_instances(
                        img=img, prob_thresh=None, nms_thresh=None, n_tiles=n_tiles
                    )[0]
                msk = msk.astype(np.uint16)
                # Compute submask area
                submsks_area = workflow.count_submsks_pixels(msk=msk)
                # Convert area in pixel^2 to µm^2
                submsks_area_um2 = {
                    k1: v1 * px_area_um2 for k1, v1 in submsks_area.items()
                }
            else:
                msk = self.data[v]["msk"]
                submsks_area = self.data[v]["submsks_area"]
                submsks_area_um2 = self.data[v]["submsks_area_um2"]

            # Discard small submasks
            filt_msk = workflow.discard_small_submsks(
                msk=msk,
                pix_dict=submsks_area_um2,
                min_n_pix=self.config["channels"][k]["min_area_um2"],
            )
            filt_msk = filt_msk.astype(np.uint16)
            output[v] = {
                "msk": msk,
                "submsks_area": submsks_area,
                "submsks_area_um2": submsks_area_um2,
                "filt_msk": filt_msk,
            }

            # Record if filtered mask has changed
            if self.do_predict:
                filt_msk_changed[v] = True
            else:
                prev_filt_msk = self.data[v]["filt_msk"]
                filt_msk_changed[v] = (
                    False if np.array_equal(filt_msk, prev_filt_msk) else True
                )

        return {"output": output, "changed": filt_msk_changed}


# %% ApplyEditsWorker() ----


class ApplyEditsWorker(BaseWorker):

    def __init__(
        self,
        copied_masks: dict,
        copied_shapes: dict,
        data: dict,
        metadata: dict,
        config: dict,
        ch_names: dict,
        parent=None,
    ):

        super().__init__(parent)
        self.copied_masks = copied_masks
        self.copied_shapes = copied_shapes
        self.data = data
        self.metadata = metadata
        self.config = config
        self.ch_names = ch_names

    def compute(self):

        px_area_um2 = self.metadata["image"]["px_area_um2"]
        output = {}

        for v in self.ch_names.values():
            ch_msk = self.copied_masks[v]
            ch_shapes = self.copied_shapes[v]
            # If user drew cell shapes, add them to the edited cell predictions
            if len(ch_shapes) > 0:
                start_idx = int(np.max(ch_msk) + 1)
                ch_msk = workflow.append_shapes_to_msk(
                    msk=ch_msk, shapes=ch_shapes, start_idx=start_idx
                )
                ch_msk = ch_msk.astype(np.uint16)
            # Drop degenerate cells with no valid contour (e.g. 1-px remnants left
            # by erasing or overwriting a cell). msk_to_cnts already excludes them,
            # but the overlap status and counts scan every non-zero id, so leaving
            # them in edit_msk would inflate collection_summary above the number of
            # ROIs actually drawn/selectable/exportable. Removing them here keeps
            # edit_msk, edit_cnts, areas, status and counts agreeing on the ROI set.
            ch_cnt = workflow.msk_to_cnts(msk=ch_msk)
            degenerate_ids = [
                i for i in np.unique(ch_msk) if i != 0 and int(i) not in ch_cnt
            ]
            if degenerate_ids:
                ch_msk = np.where(np.isin(ch_msk, degenerate_ids), 0, ch_msk).astype(
                    np.uint16
                )
            # Recompute areas from the final edited mask so they always match
            # edit_msk. Editing can delete predicted cells (freeing their ids for
            # reuse by manual ROIs, which append at start_idx = max+1) or erase
            # parts of cells; trusting the predicted areas would otherwise leave
            # stale denominators and corrupt the overlap percentages.
            ch_submsks_area = workflow.count_submsks_pixels(msk=ch_msk)
            ch_submsks_area_um2 = {
                i: j * px_area_um2 for i, j in ch_submsks_area.items()
            }
            output[v] = {
                "edit_msk": ch_msk,
                "edit_cnts": ch_cnt,
                "submsks_area": ch_submsks_area,
                "submsks_area_um2": ch_submsks_area_um2,
            }

        # The merge image is built by OverlapWorker (its only consumer); this worker
        # is concerned only with the edited masks.
        return output


# %% OverlapWorker() ----


class OverlapWorker(BaseWorker):

    def __init__(self, data: dict, config: dict, ch_names: dict, parent=None):

        super().__init__(parent)
        self.data = data
        self.config = config
        self.ch_names = ch_names

    def compute(self):

        ch0_nm = self.ch_names[0]
        ch1_nm = self.ch_names[1]
        output = {}

        # Compute ROI status
        status_ch0, status_ch1 = workflow.get_submsks1_submsks2_status(
            msk1=self.data[ch0_nm]["edit_msk"],
            msk2=self.data[ch1_nm]["edit_msk"],
            min_pct_ovl_1by2=self.config["min_pct_ovl_ch0_by_ch1"],
            min_pct_ovl_2by1=self.config["min_pct_ovl_ch1_by_ch0"],
            submsks1_pix_dict=self.data[ch0_nm]["submsks_area"],
            submsks2_pix_dict=self.data[ch1_nm]["submsks_area"],
        )

        # For each channel...
        for i, k in [(ch0_nm, status_ch0), (ch1_nm, status_ch1)]:
            # Produce cell status summary
            collection_summary = {x: len(y) for x, y in k.items()}
            collection_summary["total"] = sum(collection_summary.values())
            # Extract contours per status from the precomputed contour dict
            cnts = workflow.status_dict_to_cnts(
                status_dict=k,
                cnt_dict=self.data[i]["edit_cnts"],
            )
            output[i] = {
                "roi_status_data": k,
                "collection_summary": collection_summary,
                "cnts": cnts,
            }

        # Resolve each population's color once, then delegate the merge-image
        # rendering and ROI-ID overlay construction to workflow.py.
        def rgb_from_config(key):
            return tuple(
                int(x * 255) for x in to_rgba(self.config["elements"][key]["color"])[:3]
            )

        cnts_by_ch_idx = {0: output[ch0_nm]["cnts"], 1: output[ch1_nm]["cnts"]}
        pop_colors_rgb = {p.key: rgb_from_config(p.key) for p in POPULATIONS}
        pop_colors_rgba = {
            key: tuple(c / 255.0 for c in rgb) + (1.0,)
            for key, rgb in pop_colors_rgb.items()
        }

        # Base merge depends only on the normalized channel images + channel colors,
        # which are fixed once the image is loaded. Build it once and reuse it across
        # overlap re-runs (e.g. Adjust overlap filter) instead of re-blending.
        merge_norm_img = self.data.get("merge", {}).get("merge_norm_img")
        if merge_norm_img is None:
            merge_norm_img = workflow.make_merge_norm_img(
                img0=self.data[ch0_nm]["norm_img"],
                img1=self.data[ch1_nm]["norm_img"],
                rgba0=to_rgba(self.config["channels"][0]["color"]),
                rgba1=to_rgba(self.config["channels"][1]["color"]),
            )

        # Draw ROIs color-coded by status onto the base merge
        output["merge"] = {
            "merge_norm_img": merge_norm_img,
            "merge_norm_img_status": workflow.draw_status_contours(
                merge_norm_img=merge_norm_img,
                cnts_by_ch_idx=cnts_by_ch_idx,
                pop_colors_rgb=pop_colors_rgb,
            ),
        }

        # Centroids, rank labels and colors for the ROI IDs Points layer
        output["roi_graphics"] = workflow.build_roi_graphics(
            cnts_by_ch_idx=cnts_by_ch_idx,
            pop_colors_rgba=pop_colors_rgba,
        )

        return output


# %% _make_color_overlay_delegate() ----


def _make_color_overlay_delegate(base_cls):

    class ColorOverlayDelegate(base_cls):

        def __init__(self, layer_colors):
            super().__init__()
            self._layer_colors = layer_colors  # {layer_name: QColor}

        def initStyleOption(self, option, index):
            super().initStyleOption(option, index)
            option.displayAlignment = Qt.AlignLeft | Qt.AlignVCenter

        def paint(self, painter, option, index):
            super().paint(painter, option, index)
            name = index.data(Qt.ItemDataRole.DisplayRole)
            is_selected = bool(option.state & QStyle.State_Selected)
            if name in self._layer_colors and not is_selected:
                painter.save()
                painter.fillRect(option.rect, self._layer_colors[name])
                painter.restore()

    return ColorOverlayDelegate


# %% PluginManager() ----


class PluginManager(QWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__()

        self.viewer = viewer

        self.models = None

        self.config = copy.deepcopy(config)

        self.state = ControlState()

        self.workflow = WorkflowData()

        self.ui = UIComponents()

        self._manual_selections = {}

        self._tube_match_warning = None

        self._roi_id_base_text_size = 10
        self._roi_id_ref_zoom = None

        # Last clip-pct values used for normalization, keyed by channel index.
        # Used to detect which channels changed before a "Renormalize images" run.
        self._last_norm_pcts = {}
        self._pending_predict = False

        # Last collection summary printed to the terminal; re-prints only on change.
        self._last_collection_summary_print = None

        # Names of the models currently held in self.models; used to skip
        # reloading when the user clicks "Load models" without changing the selection.
        self._loaded_model_names: dict[int, str | None] = {0: None, 1: None}
        self._return_to_predict_after_model_change: bool = False

        # Install thread-local suppressor on sys.stdout/stderr so LoadModelWorker
        # and PredictWorker can silence StarDist/TF output in their own thread
        # without affecting the main thread or other concurrent threads.
        for _attr in ("stdout", "stderr"):
            if not isinstance(getattr(sys, _attr), _ThreadSuppressor):
                setattr(sys, _attr, _ThreadSuppressor(getattr(sys, _attr)))

        self._build_layout()

        # Pre-warm TF/stardist in the background so the first "Load models" click
        # is fast: by the time the user interacts, TF is initialized and the
        # default models are cached — the skip-reload path fires instantly.
        self._start_worker_thread(
            worker_class=LoadModelWorker,
            worker_args=(copy.deepcopy(self.config),),
            success_handler=self._on_startup_models_preloaded,
            thread_attr_name="_preload_model_worker_thread",
            worker_attr_name="_preload_model_worker",
            error_handler=lambda msg: print(f"[bruce] Background model preload failed: {msg}"),
        )

        self._init_viewer_layers()

        self._install_layer_colors()

        self._install_layer_delete_guard()

        self._remove_layer_buttons()

        self.viewer.camera.events.zoom.connect(self._update_roi_id_text_size)

        self._set_workflow_state(WorkflowState.SELECT_MODEL)

    def _install_layer_colors(self):
        qt_list = next(
            (
                w
                for w in QApplication.instance().allWidgets()
                if type(w).__name__ == "QtLayerList"
            ),
            None,
        )
        if qt_list is None:
            return

        DelegateClass = _make_color_overlay_delegate(type(qt_list.itemDelegate()))
        self._layer_delegate = DelegateClass(self._build_layer_colors())
        qt_list.setItemDelegate(self._layer_delegate)
        if hasattr(self._layer_delegate, "loading_frame_changed"):
            self._layer_delegate.loading_frame_changed.connect(
                qt_list.viewport().update
            )

        for i in [0, 1]:
            for key in ("image", "labels", "shapes"):
                self.layers[i][key].events.name.connect(self._refresh_layer_colors)

    def _build_layer_colors(self):
        layer_colors = {}
        for i in [0, 1]:
            qcolor = QColor(self.config["channels"][i]["color"])
            qcolor.setAlpha(30)
            for key in ("image", "labels", "shapes"):
                layer_colors[self.layers[i][key].name] = qcolor
        return layer_colors

    def _refresh_layer_colors(self, _event=None):
        if hasattr(self, "_layer_delegate"):
            self._layer_delegate._layer_colors = self._build_layer_colors()

    def _install_layer_delete_guard(self):
        # Prevent the user from accidentally deleting Bruce's managed layers.
        # napari routes every layer-list deletion (Delete/Backspace key and the
        # viewer's trash button) through LayerList.remove_selected(). Bruce owns
        # every layer in the viewer and reuses them across images, so we wrap
        # that method to veto removal whenever a managed layer is selected.
        layers = self.viewer.layers
        protected = {layer for layer, _ in self._iter_control_specs()}
        original_remove_selected = layers.remove_selected

        def guarded_remove_selected():
            if protected & set(layers.selection):
                self.viewer.status = "These layers can't be deleted."
                return
            original_remove_selected()

        layers.remove_selected = guarded_remove_selected

    def _remove_layer_buttons(self):
        # Hide the new-layer and delete buttons above the layer list. Bruce
        # manages a fixed set of layers, so the user should never add a new
        # Points/Shapes/Labels layer or delete an existing one.
        button_frame = next(
            (
                w
                for w in QApplication.instance().allWidgets()
                if type(w).__name__ == "QtLayerButtons"
            ),
            None,
        )
        if button_frame is None:
            return

        for attr in (
            "newPointsButton",
            "newShapesButton",
            "newLabelsButton",
            "deleteButton",
        ):
            button = getattr(button_frame, attr, None)
            if button is not None:
                button.hide()

    def closeEvent(self, event):

        self._cleanup_all_workers()

        super().closeEvent(event)

    def _cleanup_all_workers(self):

        thread_attr_names = [
            "_preload_model_worker_thread",
            "_load_model_worker_thread",
            "_load_worker_thread",
            "_normalization_worker_thread",
            "_predict_filter_worker_thread",
            "_apply_edits_worker_thread",
            "_overlap_worker_thread",
        ]

        for i in thread_attr_names:

            t: QThread | None = getattr(self, i, None)

            if isinstance(t, QThread) and t.isRunning():

                t.quit()
                t.wait()

    def _start_worker_thread(
        self,
        worker_class,
        worker_args: tuple,
        success_handler,
        thread_attr_name: str,
        worker_attr_name: str,
        allow_if_running: bool = False,
        error_handler=None,
    ) -> QThread | None:

        existing_thread = getattr(self, thread_attr_name, None)

        if existing_thread and existing_thread.isRunning():

            if not allow_if_running:

                print(
                    f"[start_worker_thread] {worker_class.__name__} not started: {thread_attr_name} still running."
                )

                return None

        thread = QThread()

        worker = worker_class(*worker_args)

        worker.moveToThread(thread)

        thread.started.connect(worker.run)

        worker.sig_success.connect(success_handler)

        default_error_handler = lambda msg, worker=worker_class.__name__: self._on_worker_error(worker, msg)
        worker.sig_error.connect(error_handler if error_handler is not None else default_error_handler)

        worker.sig_success.connect(thread.quit)
        worker.sig_error.connect(thread.quit)

        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        def reset_attrs():

            setattr(self, thread_attr_name, None)
            setattr(self, worker_attr_name, None)

        thread.finished.connect(reset_attrs)

        setattr(self, thread_attr_name, thread)
        setattr(self, worker_attr_name, worker)

        thread.start()

        return thread

    def _on_worker_error(self, worker_name: str, error_msg: str):

        self.workflow.channel_mismatch = False
        self._pending_predict = False
        self._return_to_predict_after_model_change = False

        self._set_message(
            f"{worker_name} failed with the following error:\n\n{error_msg}",
            MessageLevel.ERROR,
        )

        if self.ui.btn_clear is not None:
            self.ui.btn_clear.setEnabled(True)
        elif self.ui.btn_load_models is not None:
            # Model loading failed before any Clear button exists; re-enable the
            # model selection UI so the user can correct the selection and retry.
            if self.ui.combo_model_ch0 is not None:
                self.ui.combo_model_ch0.setEnabled(True)
            if self.ui.combo_model_ch1 is not None:
                self.ui.combo_model_ch1.setEnabled(True)
            self.ui.btn_load_models.setEnabled(True)

        return

    def _build_layout(self):

        self.file_label = QLabel("")
        self.file_label.setWordWrap(True)
        self.file_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.file_label.setStyleSheet("font-weight: bold;")
        self.file_label.setVisible(False)

        self.msg_icon = QLabel()
        self.msg_icon.setFixedSize(16, 16)
        self.msg_icon.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.msg_icon.setVisible(False)

        # Pre-render every message icon so Metal compiles the required shaders at
        # startup rather than on first use (which would log "AGX: exceeded compiled
        # variants footprint limit" if the first icon rendered is e.g. WARNING).
        style = QApplication.style()
        self._icon_pixmaps = {
            level: style.standardIcon(level.value).pixmap(16, 16)
            for level in MessageLevel
            if level.value is not None
        }

        self.msg_text = QLabel("")
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
        self.header_layout.addSpacing(30)
        self.header_layout.addWidget(self.file_label)
        self.header_layout.addWidget(self.msg_container)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.header_layout)
        self.layout.addStretch(1)

        # Host all panel content in a scroll area so its height can never force
        # napari to resize the GL canvas. On macOS native full-screen such a resize
        # wedges the window's buffer-swap/present (the panel stops painting until
        # full-screen is exited — see known_issue_macos_fullscreen_stall). Scrolling
        # internally keeps the dock/canvas size fixed regardless of panel content.
        # self.layout (the attribute used by every insert/remove call) stays the
        # inner VBox; only its host changes from the widget itself to 'content'.
        content = QWidget()
        content.setLayout(self.layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        self.setMinimumWidth(300)
        self.setMaximumWidth(1000)

    def _init_viewer_layers(self):

        self.layers = {}

        image_array = np.zeros((1, 1), dtype=np.uint8)
        label_array = np.zeros((1, 1), dtype=np.uint16)
        merge_array = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.viewer.layers.events.blocker():

            for i in [0, 1]:

                self.layers[i] = {}

                self.layers[i]["image"] = self.viewer.add_image(
                    image_array, name=f"\u25ce ch{i} image", visible=False
                )

            for i in [0, 1]:

                color_dict = {
                    0: (0.0, 0.0, 0.0, 0.0),
                    None: to_rgba(self.config["channels"][i]["color"]),
                }

                self.layers[i]["labels"] = self.viewer.add_labels(
                    label_array,
                    name=f"\u2718 ch{i} masks",
                    opacity=0.2,
                    colormap=color_dict,
                    visible=True,
                    rendering="iso_categorical",
                )

                self.layers[i]["labels"].visible = False
                self.layers[i]["labels"].contrast_limits = (0, 65535)
                self.layers[i]["labels"].brush_size = 80

            for i in [0, 1]:

                self.layers[i]["shapes"] = self.viewer.add_shapes(
                    data=[],
                    name=f"\u25b6 add ch{i}",
                    shape_type="path",
                    edge_color=self.config["channels"][i]["color"],
                    edge_width=6,
                    visible=False,
                )

            self.layers["merge"] = self.viewer.add_image(
                merge_array, name="Merge + ROIs", rgb=True, visible=False
            )

            self.layers["rois"] = self.viewer.add_points(
                np.zeros((0, 2)),
                size=0,
                name="ROI IDs",
                visible=False,
            )

        # Capture the freshly created layer-control values so that Clear can
        # restore them, undoing any change the user made in napari's "layer
        # controls" panel.
        self._snapshot_layer_controls()

    # Layer-control properties the user can change via napari's "layer controls"
    # panel. Snapshotted at startup and restored on Clear so each new image
    # starts from identical defaults. (Image contrast_limits is intentionally
    # excluded: restoring an explicit value would disable autofit, so it is
    # re-fitted from the data on image load instead.)
    _IMAGE_CONTROLS = ("opacity", "gamma", "colormap", "blending", "interpolation2d")
    _LABELS_CONTROLS = (
        "opacity",
        "colormap",
        "blending",
        "contour",
        "brush_size",
        "rendering",
        "contrast_limits",
    )
    _SHAPES_CONTROLS = (
        "opacity",
        "blending",
        "edge_width",
        "current_edge_width",
        "current_edge_color",
        "current_face_color",
    )
    _POINTS_CONTROLS = ("opacity", "blending", "size", "border_width")

    def _iter_control_specs(self):
        for i in [0, 1]:
            yield self.layers[i]["image"], self._IMAGE_CONTROLS
            yield self.layers[i]["labels"], self._LABELS_CONTROLS
            yield self.layers[i]["shapes"], self._SHAPES_CONTROLS
        yield self.layers["merge"], self._IMAGE_CONTROLS
        yield self.layers["rois"], self._POINTS_CONTROLS

    def _snapshot_layer_controls(self):
        self._layer_control_defaults = []
        for layer, names in self._iter_control_specs():
            snap = {}
            for name in names:
                if not hasattr(layer, name):
                    continue
                value = getattr(layer, name)
                # Copy mutable values (lists / arrays) so a later in-place edit
                # cannot corrupt the stored default; keep evented objects (e.g.
                # colormap) by reference since napari replaces rather than
                # mutates them.
                if isinstance(value, (list, np.ndarray)):
                    value = np.array(value, copy=True)
                snap[name] = value
            self._layer_control_defaults.append((layer, snap))

    def _restore_layer_controls(self):
        for layer, snap in self._layer_control_defaults:
            for name, value in snap.items():
                try:
                    setattr(layer, name, value)
                except Exception:
                    # Be tolerant of napari version differences in control
                    # setters; a single failed control must not abort Clear.
                    pass

    def _reset_viewer_layers(self):

        image_array = np.zeros((1, 1), dtype=np.uint8)
        label_array = np.zeros((1, 1), dtype=np.uint16)
        merge_array = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.viewer.layers.events.blocker():

            # Restore every layer-control value (opacity, gamma, colormap,
            # contour, brush size, edge color/width, ...) to the startup
            # defaults so user changes in napari's layer controls do not carry
            # over to the next image.
            self._restore_layer_controls()

            for i in [0, 1]:

                self.layers[i]["image"].data = image_array
                self.layers[i]["image"].name = f"\u25ce ch{i} image"

                self.layers[i]["labels"].visible = False
                self.layers[i]["labels"].data = label_array
                self.layers[i]["labels"].name = f"\u2718 ch{i} masks"

                self.layers[i]["shapes"].visible = False
                self.layers[i]["shapes"].data = []
                self.layers[i]["shapes"].name = f"\u25b6 add ch{i}"

            self.layers["merge"].data = merge_array
            self.layers["merge"].name = "Merge + ROIs"

            # Reset text to a scalar constant before clearing the points so no
            # stale per-point text array (trimmed to the old point count) lingers
            # to be indexed out of bounds on the next image's overlap pass.
            self.layers["rois"].text = {"constant": ""}
            self.layers["rois"].data = np.zeros((0, 2))
            self.layers["rois"].visible = False

    def _set_workflow_state(self, state: WorkflowState, force=False):

        if not force and state == self.state.workflow_state:

            return

        self.state.workflow_state = state

        self._update_viewer_from_workflow()
        self._update_ui_from_workflow()

    def _update_viewer_from_workflow(self):

        workflow_state = self.state.workflow_state

        # Safe default; every known workflow state is handled by a branch below.
        viewer_state = ViewerState.LOCKED

        if workflow_state in {
            WorkflowState.SELECT_MODEL,
            WorkflowState.SELECT_FILE,
            WorkflowState.LOAD_IMAGE,
            WorkflowState.LOADING_IMAGE,
            WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI,
        }:

            viewer_state = ViewerState.NO_IMAGE

        elif workflow_state in {
            WorkflowState.IMAGE_LOADED_FOR_PREDICT_ROI,
            WorkflowState.PREDICT_ROI,
            WorkflowState.PREDICTING_ROI,
            # Apply-edits and overlap run as background computations: show the
            # plain channel images (merge hidden) until the status-colored merge
            # is ready, so neither a stale nor a placeholder merge is shown.
            WorkflowState.APPLYING_EDITS,
            WorkflowState.OVERLAPPING_ROI,
        }:

            viewer_state = ViewerState.IMAGE_LOADED

        elif workflow_state in {WorkflowState.APPLY_EDITS}:

            viewer_state = ViewerState.ROI_EDITING

        elif workflow_state in {
            WorkflowState.OVERLAP_FILTER_OR_SAVE,
            WorkflowState.UPDATE_OVERLAP_FILTER_OR_SAVE,
            WorkflowState.CLEARED,
        }:

            viewer_state = ViewerState.LOCKED

        self._set_viewer_state(state=viewer_state)

    def _set_viewer_state(self, state: ViewerState):

        if state == self.state.viewer_state:

            return

        self.state.viewer_state = state

        with self.viewer.layers.events.blocker():

            # Baseline state
            for i in [0, 1]:

                for j in ["image", "labels", "shapes"]:

                    self.layers[i][j].visible = False

                for j in ["labels", "shapes"]:

                    self.layers[i][j].editable = False

            self.layers["merge"].visible = False
            self.layers["rois"].visible = False

            # No image
            if state == ViewerState.NO_IMAGE:

                return

            # Image loaded
            if state == ViewerState.IMAGE_LOADED:

                for k in [0, 1]:

                    self.layers[k]["image"].visible = True

                self.viewer.layers.selection.active = self.layers[1]["image"]

            # ROI editing allowed
            elif state == ViewerState.ROI_EDITING:

                for k in [0, 1]:

                    self.layers[k]["image"].visible = True

                    labels = self.layers[k]["labels"]

                    with labels.events.blocker():

                        labels.editable = True
                        labels.visible = True
                        labels.mode = "erase"

                    # labels.visible = True

                    shapes = self.layers[k]["shapes"]

                    with shapes.events.blocker():

                        shapes.editable = True
                        shapes.visible = True
                        shapes.mode = "add_path"

                    # shapes.visible = True

                self.viewer.layers.selection.active = self.layers[1]["shapes"]

            # ROI editing disabled
            elif state == ViewerState.LOCKED:

                for k in [0, 1]:

                    self.layers[k]["image"].visible = True

                self.layers["merge"].visible = True

                self.viewer.layers.selection.active = self.layers["merge"]

    def _update_ui_from_workflow(self):

        workflow_state = self.state.workflow_state

        if workflow_state == WorkflowState.SELECT_MODEL:

            self._set_select_model_ui()

        elif workflow_state == WorkflowState.SELECT_FILE:

            self._set_select_file_ui()

        elif workflow_state == WorkflowState.LOAD_IMAGE:

            self._set_load_image_ui()

        elif workflow_state == WorkflowState.PREDICT_ROI:

            self._set_predict_roi_ui(loaded_for_predict=False)

        elif workflow_state == WorkflowState.IMAGE_LOADED_FOR_PREDICT_ROI:

            self._set_predict_roi_ui(loaded_for_predict=True)

        elif workflow_state == WorkflowState.PREDICTING_ROI:

            self._set_predicting_roi_ui()

        elif workflow_state == WorkflowState.APPLY_EDITS:

            self._set_apply_edits_ui()

        elif workflow_state == WorkflowState.APPLYING_EDITS:

            self._set_applying_edits_ui()

        elif workflow_state == WorkflowState.OVERLAP_FILTER_OR_SAVE:

            self._set_overlap_filter_or_save_ui(build_ui=True)

        elif workflow_state == WorkflowState.UPDATE_OVERLAP_FILTER_OR_SAVE:

            self._set_overlap_filter_or_save_ui(build_ui=False)

        elif workflow_state == WorkflowState.CLEARED:

            self._set_cleared_ui()

        if workflow_state in {
            WorkflowState.LOADING_IMAGE,
            WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI,
            WorkflowState.PREDICTING_ROI,
            WorkflowState.APPLYING_EDITS,
            WorkflowState.OVERLAPPING_ROI,
        }:

            self._set_ui_enabled(False)

    def _set_ui_enabled(self, enabled: bool):

        for i in vars(self.ui).values():

            if isinstance(i, QWidget):

                i.setEnabled(enabled)

        for btn in self.ui.btns_choose_rois.values():

            btn.setEnabled(enabled)

    def _destroy_ui_widgets(self, names):
        """Remove, hide and delete the named self.ui widgets, resetting each to None.

        Missing/None widgets are skipped, so this is safe to call on partially
        built panels.
        """

        for name in names:

            widget = getattr(self.ui, name, None)

            if widget is not None:

                self.layout.removeWidget(widget)
                widget.hide()
                widget.deleteLater()
                setattr(self.ui, name, None)

    def _destroy_elem_boxes(self):
        """Tear down the per-population element boxes and drop their references.

        The 'Choose ROIs' buttons live inside the element boxes, so they are
        destroyed together with them; only their references need dropping.
        """

        for box in self.ui.box_elems.values():

            self.layout.removeWidget(box)
            box.hide()
            box.deleteLater()

        self.ui.box_elems.clear()
        self.ui.btns_choose_rois.clear()

    def _set_select_model_ui(self):

        model_options = list(configuration.list_stardist_models().keys())

        ch_labels = {
            0: f"{self.workflow.ch_names[0]} model" if self.workflow.ch_names else "ch0 model",
            1: f"{self.workflow.ch_names[1]} model" if self.workflow.ch_names else "ch1 model",
        }

        row0 = QWidget()
        row0_layout = QHBoxLayout(row0)
        row0_layout.setContentsMargins(0, 0, 0, 0)
        row0_layout.setSpacing(4)
        row0_layout.addWidget(QLabel(ch_labels[0]))
        combo0 = QComboBox()
        combo0.addItems(model_options)
        combo0.setCurrentText(self.config["channels"][0]["stardist_model"])
        row0_layout.addWidget(combo0, stretch=1)
        self.ui.box_model_ch0 = row0
        self.ui.combo_model_ch0 = combo0

        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(4)
        row1_layout.addWidget(QLabel(ch_labels[1]))
        combo1 = QComboBox()
        combo1.addItems(model_options)
        combo1.setCurrentText(self.config["channels"][1]["stardist_model"])
        row1_layout.addWidget(combo1, stretch=1)
        self.ui.box_model_ch1 = row1
        self.ui.combo_model_ch1 = combo1

        btn_load_models = QPushButton("Load models")
        btn_load_models.clicked.connect(self._on_load_models_clicked)
        self.ui.btn_load_models = btn_load_models

        for i, w in enumerate([row0, row1, btn_load_models]):
            self.layout.insertWidget(i, w)

    def _set_select_file_ui(self):

        # Tear down model selection UI (present when coming from SELECT_MODEL).
        self._destroy_ui_widgets(["box_model_ch0", "combo_model_ch0", "box_model_ch1", "combo_model_ch1", "btn_load_models"])

        btn_select = QPushButton("Select file")
        btn_select.clicked.connect(self._on_select_clicked)
        self.header_layout.insertWidget(0, btn_select)
        self.ui.btn_select = btn_select

        # "Change models" at the top of the panel, above the "Select file" row.
        btn_change_models = QPushButton("Change models")
        btn_change_models.clicked.connect(self._on_change_models_clicked)
        self.ui.btn_change_models = btn_change_models
        self.layout.insertWidget(0, btn_change_models)

    def _set_load_image_ui(self):

        # Remove 'Select file' button from viewer
        self._destroy_ui_widgets(["btn_select"])

        # Add 'Change models' button at position 0.
        # Guard: when entering LOAD_IMAGE from SELECT_FILE (user clicked "Select file"),
        # the button was already created by _set_select_file_ui — don't create a second one.
        if self.ui.btn_change_models is None:
            btn_change_models = QPushButton("Change models")
            btn_change_models.clicked.connect(self._on_change_models_clicked)
            self.ui.btn_change_models = btn_change_models
            self.layout.insertWidget(0, btn_change_models)

        # Add clip-percentile boxes (positions 1-4) above the action buttons
        self._add_clip_pct_boxes(start=1)

        # Add 'Load images', 'Predict cells' and 'Clear' buttons below the boxes
        btn_load = QPushButton("Load images")
        btn_load.clicked.connect(self._on_load_clicked)
        self.ui.btn_load = btn_load

        btn_predict = QPushButton("Predict cells")
        btn_predict.clicked.connect(self._on_predict_clicked)
        self.ui.btn_predict = btn_predict

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda checked=False: self._on_clear_clicked())
        self.ui.btn_clear = btn_clear

        for i, j in zip([5, 6, 7], [btn_load, btn_predict, btn_clear]):

            self.layout.insertWidget(i, j)

    def _add_min_area_boxes(self, insert_pos: int = 0):

        # Add 'min n pix' boxes to viewer
        box_min_area_ch0 = ParamValueBox(
            label=f"min area (\u00b5m\u00b2) {self.workflow.ch_names[0]}",
            default=self.config["channels"][0]["min_area_um2"],
            min_val=0.0,
            max_val=10000.0,
        )
        box_min_area_ch0.valueChanged.connect(lambda x: self._on_min_area_changed(0, x))
        self.ui.box_min_area_ch0 = box_min_area_ch0

        box_min_area_ch1 = ParamValueBox(
            label=f"min area (\u00b5m\u00b2) {self.workflow.ch_names[1]}",
            default=self.config["channels"][1]["min_area_um2"],
            min_val=0.0,
            max_val=10000.0,
        )
        box_min_area_ch1.valueChanged.connect(lambda x: self._on_min_area_changed(1, x))
        self.ui.box_min_area_ch1 = box_min_area_ch1

        for i, j in zip([insert_pos, insert_pos + 1], [box_min_area_ch0, box_min_area_ch1]):

            self.layout.insertWidget(i, j)

    def _update_clip_pct_box_labels(self):
        """Refresh clip-pct box labels to show real channel names.

        Called after ch_names are known (image load) and when rebuilding the
        PREDICT_ROI panel with freshly created boxes.
        """
        for attr, label in [
            ("box_low_pct_ch0", f"low % clip - {self.workflow.ch_names[0]}"),
            ("box_high_pct_ch0", f"high % clip - {self.workflow.ch_names[0]}"),
            ("box_low_pct_ch1", f"low % clip - {self.workflow.ch_names[1]}"),
            ("box_high_pct_ch1", f"high % clip - {self.workflow.ch_names[1]}"),
        ]:
            box = getattr(self.ui, attr, None)
            if box is not None:
                box._label.setText(label)

    def _add_clip_pct_boxes(self, start: int = 0):

        box_low_pct_ch0 = ParamValueBox(
            label="low % clip - ch0",
            default=self.config["channels"][0]["low_pct"],
            min_val=0.0,
            max_val=100.0,
            decimals=4,
        )
        box_low_pct_ch0.valueChanged.connect(
            lambda x: self._on_clip_pct_changed(0, "low_pct", x)
        )
        self.ui.box_low_pct_ch0 = box_low_pct_ch0

        box_high_pct_ch0 = ParamValueBox(
            label="high % clip - ch0",
            default=self.config["channels"][0]["high_pct"],
            min_val=0.0,
            max_val=100.0,
            decimals=4,
        )
        box_high_pct_ch0.valueChanged.connect(
            lambda x: self._on_clip_pct_changed(0, "high_pct", x)
        )
        self.ui.box_high_pct_ch0 = box_high_pct_ch0

        box_low_pct_ch1 = ParamValueBox(
            label="low % clip - ch1",
            default=self.config["channels"][1]["low_pct"],
            min_val=0.0,
            max_val=100.0,
            decimals=4,
        )
        box_low_pct_ch1.valueChanged.connect(
            lambda x: self._on_clip_pct_changed(1, "low_pct", x)
        )
        self.ui.box_low_pct_ch1 = box_low_pct_ch1

        box_high_pct_ch1 = ParamValueBox(
            label="high % clip - ch1",
            default=self.config["channels"][1]["high_pct"],
            min_val=0.0,
            max_val=100.0,
            decimals=4,
        )
        box_high_pct_ch1.valueChanged.connect(
            lambda x: self._on_clip_pct_changed(1, "high_pct", x)
        )
        self.ui.box_high_pct_ch1 = box_high_pct_ch1

        for i, j in zip(
            [start, start + 1, start + 2, start + 3],
            [box_low_pct_ch0, box_high_pct_ch0, box_low_pct_ch1, box_high_pct_ch1],
        ):
            self.layout.insertWidget(i, j)

    def _set_predict_roi_ui(self, loaded_for_predict=False):

        # Insert min-area boxes between "Renormalize images" (index 5) and "Predict cells"
        # Index 5 because "Change models" button is at 0, clip-pct boxes at 1-4.
        self._add_min_area_boxes(insert_pos=6)

        if loaded_for_predict:

            # Disable 'min n pix' boxes
            for i in [self.ui.box_min_area_ch0, self.ui.box_min_area_ch1]:

                i.setEnabled(False)

        else:

            # Re-enable 'Change models', clip-pct boxes, 'Renormalize images',
            # 'Predict cells' and 'Clear'
            for i in [
                self.ui.btn_change_models,
                self.ui.box_low_pct_ch0,
                self.ui.box_high_pct_ch0,
                self.ui.box_low_pct_ch1,
                self.ui.box_high_pct_ch1,
                self.ui.btn_load,
                self.ui.btn_predict,
                self.ui.btn_clear,
            ]:

                assert i is not None
                i.setEnabled(True)

    def _set_predicting_roi_ui(self):

        self._destroy_ui_widgets([
            "btn_load",
            "btn_predict",
            "box_low_pct_ch0",
            "box_high_pct_ch0",
            "box_low_pct_ch1",
            "box_high_pct_ch1",
        ])

    def _set_apply_edits_ui(self):

        # Add 'Repredict cells' at the very top, then the four edit buttons below
        # the min-area boxes.
        btn_repredict = QPushButton("Repredict cells")
        btn_repredict.clicked.connect(self._on_repredict_clicked)
        self.ui.btn_repredict = btn_repredict

        btn_filter_size = QPushButton("Adjust size filter")
        btn_filter_size.clicked.connect(self._on_filter_size_clicked)
        self.ui.btn_filter_size = btn_filter_size

        btn_discard_additions = QPushButton("Discard added ROIs")
        btn_discard_additions.clicked.connect(self._on_discard_additions_clicked)
        self.ui.btn_discard_additions = btn_discard_additions

        btn_discard_deletions = QPushButton("Restore deleted ROIs")
        btn_discard_deletions.clicked.connect(self._on_discard_deletions_clicked)
        self.ui.btn_discard_deletions = btn_discard_deletions

        btn_apply_edits = QPushButton("Apply edits")
        btn_apply_edits.clicked.connect(self._on_apply_edits_clicked)
        self.ui.btn_apply_edits = btn_apply_edits

        # "Change models" is at index 0 when entering from PREDICT_ROI/PREDICTING_ROI
        # (btn_change_models persists). When entering from "Edit ROIs" (OVERLAP_FILTER_OR_SAVE
        # → APPLY_EDITS), it was destroyed by _set_applying_edits_ui and must be re-created.
        if self.ui.btn_change_models is None:
            btn_change_models = QPushButton("Change models")
            btn_change_models.clicked.connect(self._on_change_models_clicked)
            self.ui.btn_change_models = btn_change_models
            self.layout.insertWidget(0, btn_change_models)

        # btn_repredict goes at index 1, above the two min-area boxes (2-3);
        # edit buttons follow at indices 4-7.
        self.layout.insertWidget(1, btn_repredict)
        for i, j in zip(
            [4, 5, 6, 7],
            [
                btn_filter_size,
                btn_discard_additions,
                btn_discard_deletions,
                btn_apply_edits,
            ],
        ):

            self.layout.insertWidget(i, j)

        # Re-enable 'Change models', 'min n pix' boxes, the edit buttons and 'Clear'
        for i in [
            self.ui.btn_change_models,
            self.ui.btn_repredict,
            self.ui.box_min_area_ch0,
            self.ui.box_min_area_ch1,
            self.ui.btn_filter_size,
            self.ui.btn_discard_additions,
            self.ui.btn_discard_deletions,
            self.ui.btn_apply_edits,
            self.ui.btn_clear,
        ]:

            assert i is not None
            i.setEnabled(True)

    def _set_applying_edits_ui(self):

        # Remove 'Change models', 'Repredict cells', 'min n pix' boxes and
        # the edit buttons from viewer (overlap/save UI follows, no models button there)
        self._destroy_ui_widgets(
            [
                "btn_change_models",
                "btn_repredict",
                "box_min_area_ch0",
                "box_min_area_ch1",
                "btn_filter_size",
                "btn_discard_additions",
                "btn_discard_deletions",
                "btn_apply_edits",
            ]
        )

    def _add_min_pct_ovl_boxes(self):

        # Add 'min % ovl' boxes to viewer
        box_min_pct_ovl_ch0_by_ch1 = ParamValueBox(
            label=f"min % overlap {self.workflow.ch_names[0]} / {self.workflow.ch_names[1]}",
            default=self.config["min_pct_ovl_ch0_by_ch1"],
            min_val=0.0,
            max_val=100.0,
        )
        box_min_pct_ovl_ch0_by_ch1.valueChanged.connect(
            lambda x: self._on_min_pct_ovl_changed("min_pct_ovl_ch0_by_ch1", x)
        )
        self.ui.box_min_pct_ovl_ch0_by_ch1 = box_min_pct_ovl_ch0_by_ch1

        box_min_pct_ovl_ch1_by_ch0 = ParamValueBox(
            label=f"min % overlap {self.workflow.ch_names[1]} / {self.workflow.ch_names[0]}",
            default=self.config["min_pct_ovl_ch1_by_ch0"],
            min_val=0.0,
            max_val=100.0,
        )
        box_min_pct_ovl_ch1_by_ch0.valueChanged.connect(
            lambda x: self._on_min_pct_ovl_changed("min_pct_ovl_ch1_by_ch0", x)
        )
        self.ui.box_min_pct_ovl_ch1_by_ch0 = box_min_pct_ovl_ch1_by_ch0

        return [box_min_pct_ovl_ch0_by_ch1, box_min_pct_ovl_ch1_by_ch0]

    def _set_overlap_filter_or_save_ui(self, build_ui=True):

        if build_ui:

            # Add 'min % ovl' boxes, 'Adjust overlap filter' button, 'element'
            # boxes, 'Edit ROIs' and 'Save results' buttons to viewer if overlaps
            # are returned for the first time
            box_min_pct_ovl_list = self._add_min_pct_ovl_boxes()

            btn_overlap_filter = QPushButton("Adjust overlap filter")
            btn_overlap_filter.clicked.connect(self._on_overlap_filter_clicked)
            self.ui.btn_overlap_filter = btn_overlap_filter

            btn_edit_rois = QPushButton("Edit ROIs")
            btn_edit_rois.clicked.connect(self._on_edit_rois_clicked)
            self.ui.btn_edit_rois = btn_edit_rois

            btn_save = QPushButton("Save results")
            btn_save.clicked.connect(self._on_save_clicked)
            self.ui.btn_save = btn_save

            # Build one element box per population, in the order the user listed
            # them in config["elements"] so the panel reflects their chosen order.
            box_elem_list = []
            for p in self._populations_in_config_order():
                box = ElementConfigBox(label=self._elem_label(p))
                box.valueChanged.connect(
                    lambda x, key=p.key: self._on_elem_params_changed(key, x)
                )
                # Clear manual selection when the spinner is edited directly.
                box.nCollectChanged.connect(
                    lambda n, key=p.key: self._on_n_collect_changed(key)
                )
                # Wire the box's built-in "Choose ROIs" button (it lives inside the
                # element box, so no extra rows are added to the panel).
                box.btn_choose_rois.clicked.connect(
                    lambda checked=False, key=p.key: self._on_choose_rois_clicked(key)
                )
                self.ui.box_elems[p.key] = box
                self.ui.btns_choose_rois[p.key] = box.btn_choose_rois
                box_elem_list.append(box)

            # Insert widgets: btn_edit_rois at top, then min % ovl boxes,
            # btn_overlap_filter, element boxes, btn_save
            widgets_to_insert = (
                [btn_edit_rois]
                + box_min_pct_ovl_list
                + [btn_overlap_filter]
                + box_elem_list
                + [btn_save]
            )

            for i, w in enumerate(widgets_to_insert):
                self.layout.insertWidget(i, w)

        # Re-enable 'min % ovl' and 'element' boxes, 'Adjust overlap filter', 'Save results' and 'Clear' buttons
        # => 'Adjust overlap filter' button, 'element' boxes and 'Save results' button do not exist yet if overlaps are returned for the first time
        for i in [
            self.ui.box_min_pct_ovl_ch0_by_ch1,
            self.ui.box_min_pct_ovl_ch1_by_ch0,
            self.ui.btn_overlap_filter,
            *self.ui.box_elems.values(),
            self.ui.btn_edit_rois,
            self.ui.btn_save,
            self.ui.btn_clear,
        ]:

            assert i is not None
            i.setEnabled(True)

        for btn in self.ui.btns_choose_rois.values():
            btn.setEnabled(True)

        # Update 'element' boxes
        self._update_elem_boxes()

    def _set_cleared_ui(self):

        self._destroy_ui_widgets(
            [
                "box_model_ch0",
                "combo_model_ch0",
                "box_model_ch1",
                "combo_model_ch1",
                "btn_load_models",
                "btn_change_models",
                "btn_clear",
                "btn_load",
                "btn_predict",
                "box_low_pct_ch0",
                "box_high_pct_ch0",
                "box_low_pct_ch1",
                "box_high_pct_ch1",
                "btn_repredict",
                "btn_filter_size",
                "btn_discard_additions",
                "btn_discard_deletions",
                "box_min_area_ch0",
                "box_min_area_ch1",
                "btn_apply_edits",
                "btn_overlap_filter",
                "btn_edit_rois",
                "box_min_pct_ovl_ch0_by_ch1",
                "box_min_pct_ovl_ch1_by_ch0",
                "btn_save",
            ]
        )

        self._destroy_elem_boxes()

    def _update_viewer_data_on_load_finished(self):

        with self.viewer.layers.events.blocker():

            # Add normalized images to viewer
            for k, v in self.workflow.ch_names.items():

                self.layers[k]["image"].data = self.workflow.data[v]["norm_img"]
                # Autofit contrast to the new image, discarding any manual
                # contrast the user set on a previously loaded image.
                self.layers[k]["image"].reset_contrast_limits_range()
                self.layers[k]["image"].reset_contrast_limits()
                self.layers[k]["image"].name = f"\u25ce {v} image"

                self.layers[k]["labels"].name = f"\u2718 {v} masks"

                self.layers[k]["shapes"].name = f"\u25b6 add {v}"

            self.layers["merge"].data = np.zeros(
                self.workflow.data[self.workflow.ch_names[0]]["norm_img"].shape + (3,),
                dtype=np.uint8,
            )

        self.viewer.reset_view()

        # Anchor the ROI ID label size to the fit-to-view zoom captured HERE,
        # immediately after reset_view, rather than later at overlap-finish.
        # reset_view sets camera.zoom = 0.95 * min(canvas_size / image_size), so the
        # zoom depends on the canvas size at this instant. Capturing it now — when it
        # provably matches the canvas the image was just fit to — avoids a race where a
        # canvas resize between load and overlap-finish (e.g. entering full-screen)
        # inflates camera.zoom and makes the labels render far too small.
        self._roi_id_ref_zoom = self.viewer.camera.zoom

    def _update_viewer_data_on_predict_filter_finished(self, filt_msk_changed: dict):

        for k, v in self.workflow.ch_names.items():

            if filt_msk_changed[v]:

                labels = self.layers[k]["labels"]

                with labels.events.blocker():

                    # Hand the layer a copy so in-place edits (erasing submasks)
                    # don't mutate the stored filt_msk: it stays the pure
                    # size-filtered prediction. User edits are captured separately
                    # into edit_msk at apply-edits time.
                    labels.data = self.workflow.data[v]["filt_msk"].copy()

    def _update_viewer_data_on_overlap_finished(self):

        roi_graphics = self.workflow.data.get("roi_graphics", {})
        ranked_ids = roi_graphics.get("ranked_ids", [])
        centroids = roi_graphics.get("centroids", np.zeros((0, 2)))
        colors = roi_graphics.get("colors", np.zeros((0, 4)))

        self.layers["merge"].data = self.workflow.data["merge"]["merge_norm_img_status"]

        pts = centroids if len(centroids) > 0 else np.zeros((0, 2))
        layer = self.layers["rois"]

        # _roi_id_ref_zoom was captured at image-load (see
        # _update_viewer_data_on_load_finished); do NOT recapture it here from the
        # current camera, which may have been zoomed by the user or rescaled by a
        # canvas resize since load.

        # Updating the ROI-ID labels is order-sensitive. napari's vispy text node
        # re-renders synchronously whenever layer.text.events fire, indexing the
        # per-point text array with the layer's _indices_view. While the layer is
        # invisible (as here, until the end), its slice is not refreshed, so
        # _indices_view stays stale at the PREVIOUS overlap pass's (larger) point
        # count. Assigning the new, shorter text/points then triggers an
        # out-of-bounds index -> IndexError and a frozen viewer (seen on the second
        # image after Clear, and on the Edit ROIs -> re-overlap round-trip).
        #
        # Fix: mutate data with the text-render callback blocked, force a slice
        # refresh so _indices_view matches the final point count, and only then
        # assign the text once against the now-consistent state.
        if len(ranked_ids) > 0:
            hex_colors = [
                "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
                for r, g, b, *_ in colors
            ]
            text_dict = {
                "string": list(ranked_ids),
                "size": self._roi_id_base_text_size,
                "color": hex_colors,
                "visible": True,
            }
        else:
            text_dict = {"constant": ""}

        with layer.text.events.blocker_all():
            layer.visible = True
            layer.data = pts
            # force=True refreshes the slice even though we just toggled visibility
            # in the same blocked window, syncing _indices_view to len(pts).
            layer.refresh(force=True)

        # data, _indices_view and the text array are now consistent: assigning text
        # here renders safely.
        layer.text = text_dict

        if len(ranked_ids) > 0:
            # The dict sets size to the raw base value, which would render at
            # base / current_zoom. Rescale it to the load-time reference zoom so the
            # initial draw uses the same fixed apparent size as every later zoom step.
            self._update_roi_id_text_size()

    def _update_roi_id_text_size(self, event=None):

        # Rescale ROI ID font with zoom so its on-screen size stays constant.
        # Text size is in data coordinates: on-screen size ~ size / zoom, so size must
        # track zoom (size = base * zoom / ref_zoom) to keep the apparent size fixed.
        if self._roi_id_ref_zoom is None:
            return

        layer = self.layers.get("rois")

        if (
            layer is None
            or len(layer.data) == 0
            or not getattr(layer, "visible", False)
        ):
            return

        zoom = self.viewer.camera.zoom

        try:
            layer.text.size = self._roi_id_base_text_size * (
                zoom / self._roi_id_ref_zoom
            )
        except Exception:
            pass

    def _set_message(self, text: str, level: MessageLevel = MessageLevel.NONE):

        if self.workflow.channel_mismatch:

            text = (
                f"{text}\n\n⚠ channel name mismatch between image and config."
            )

        self.msg_text.setText(text)

        if level.value is None:

            self.msg_icon.setVisible(False)

        else:

            self.msg_icon.setPixmap(self._icon_pixmaps[level])
            self.msg_icon.setVisible(True)

    def _show_busy_message(self, text: str):
        """Show a busy message and flush pending events so it paints before the
        (synchronous) viewer update that follows."""

        self._set_message(text, MessageLevel.BUSY)
        QApplication.processEvents()

    def _on_change_models_clicked(self):

        # Remember whether we have loaded images so we can skip loading on return.
        self._return_to_predict_after_model_change = bool(self.workflow.ch_names)

        # Tear down all widgets that may be present in SELECT_FILE, LOAD_IMAGE,
        # PREDICT_ROI, or APPLY_EDITS (safe to call on None — _destroy_ui_widgets skips them).
        self._destroy_ui_widgets([
            "btn_change_models",
            "btn_select",
            "box_low_pct_ch0",
            "box_high_pct_ch0",
            "box_low_pct_ch1",
            "box_high_pct_ch1",
            "btn_load",
            "btn_predict",
            "btn_repredict",
            "box_min_area_ch0",
            "box_min_area_ch1",
            "btn_filter_size",
            "btn_discard_additions",
            "btn_discard_deletions",
            "btn_apply_edits",
        ])

        self._set_workflow_state(WorkflowState.SELECT_MODEL)

    def _on_load_models_clicked(self):

        model0 = self.ui.combo_model_ch0.currentText()
        model1 = self.ui.combo_model_ch1.currentText()

        # Skip reloading when the selected models are already in memory.
        if (
            self.models is not None
            and self._loaded_model_names.get(0) == model0
            and self._loaded_model_names.get(1) == model1
        ):
            self._set_message("", MessageLevel.NONE)
            self._after_models_ready()
            return

        self.config["channels"][0]["stardist_model"] = model0
        self.config["channels"][1]["stardist_model"] = model1

        self._set_ui_enabled(False)

        self._set_message(
            f"Loading StarDist models...\n\nChannel 0: {model0}\nChannel 1: {model1}",
            MessageLevel.WORK,
        )

        self._start_worker_thread(
            worker_class=LoadModelWorker,
            worker_args=(self.config,),
            success_handler=self._on_models_loaded,
            thread_attr_name="_load_model_worker_thread",
            worker_attr_name="_load_model_worker",
        )

    def _on_models_loaded(self, models: dict):

        self.models = models
        self._loaded_model_names = {
            0: self.config["channels"][0]["stardist_model"],
            1: self.config["channels"][1]["stardist_model"],
        }

        self._set_message("", MessageLevel.NONE)

        self._after_models_ready()

    def _on_startup_models_preloaded(self, models: dict):
        """Silent handler for the background model preload at startup.

        Caches the result so _on_load_models_clicked can take the skip-reload
        path instantly when the user keeps the default model selection.
        Does not advance the workflow state or touch the UI.
        """
        self.models = models
        self._loaded_model_names = {
            0: self.config["channels"][0]["stardist_model"],
            1: self.config["channels"][1]["stardist_model"],
        }

    def _after_models_ready(self):
        """Transition to the correct state after models are loaded (or skipped)."""

        if self._return_to_predict_after_model_change:
            # Images were already loaded — skip file selection and loading entirely.
            self._return_to_predict_after_model_change = False
            self._destroy_ui_widgets(["box_model_ch0", "combo_model_ch0", "box_model_ch1", "combo_model_ch1", "btn_load_models"])
            self._rebuild_predict_roi_panel(add_change_models=True)
        elif self.workflow.path:
            # File was selected but images not loaded yet — skip file selection,
            # go straight back to LOAD_IMAGE. Destroy btn_clear so _set_load_image_ui
            # re-creates it cleanly (it was left over from the previous LOAD_IMAGE pass).
            self._destroy_ui_widgets(["box_model_ch0", "combo_model_ch0", "box_model_ch1", "combo_model_ch1", "btn_load_models", "btn_clear"])
            self._set_workflow_state(WorkflowState.LOAD_IMAGE)
        else:
            self._set_workflow_state(WorkflowState.SELECT_FILE)

    def _on_select_clicked(self):

        # Record user-selected file path
        self.workflow.path = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select .zvi file",
            directory=str(Path(self.config["in_dir_path"]).expanduser()),
            filter="Images (*.zvi)",
        )[0]

        # Abort if user cancelled selection
        if self.workflow.path == "":

            self._set_message(".zvi file selection cancelled", MessageLevel.WARNING)

            return

        self._set_workflow_state(state=WorkflowState.LOAD_IMAGE)

        # Display the selected file name above the button panel
        self.file_label.setText(Path(self.workflow.path).stem)
        self.file_label.setVisible(True)

        self._set_message(".zvi file selected", MessageLevel.INFO)

    def _on_clear_clicked(self, *args):

        # Reset attributes to default values
        self.config = copy.deepcopy(config)

        self._manual_selections.clear()

        self._tube_match_warning = None

        self._last_norm_pcts.clear()

        self._return_to_predict_after_model_change = False

        self._set_workflow_state(state=WorkflowState.CLEARED)

        self.workflow.reset()

        self.ui.reset()

        self.state.reset()

        self._reset_viewer_layers()

        self._set_workflow_state(state=WorkflowState.SELECT_MODEL)

        self.file_label.setText("")
        self.file_label.setVisible(False)

        self._set_message("", MessageLevel.NONE)

    def _on_load_clicked(self):

        self._set_workflow_state(state=WorkflowState.LOADING_IMAGE)

        self._set_message("Loading images...", MessageLevel.WORK)

        # Trigger load worker thread
        self._start_load_thread()

    def _get_changed_pct_channels(self) -> list[int]:
        """Return indices of channels whose clip-pct values differ from the last normalization."""
        return [
            i for i in [0, 1]
            if (
                self.config["channels"][i]["low_pct"] != self._last_norm_pcts.get(i, {}).get("low_pct")
                or self.config["channels"][i]["high_pct"] != self._last_norm_pcts.get(i, {}).get("high_pct")
            )
        ]

    def _on_reload_clicked(self):

        changed = self._get_changed_pct_channels()

        if not changed:
            self._set_message("No clip values changed — images unchanged.", MessageLevel.INFO)
            return

        self._set_ui_enabled(False)

        self._start_worker_thread(
            worker_class=NormalizationWorker,
            worker_args=(self.workflow.data, self.workflow.ch_names, self.config, changed),
            success_handler=self._on_normalization_finished,
            thread_attr_name="_normalization_worker_thread",
            worker_attr_name="_normalization_worker",
        )

    def _on_normalization_finished(self, worker_output: dict):

        for i, norm_img in worker_output.items():
            ch_nm = self.workflow.ch_names[i]
            self.workflow.data[ch_nm]["norm_img"] = norm_img
            self._last_norm_pcts[i] = {
                "low_pct": self.config["channels"][i]["low_pct"],
                "high_pct": self.config["channels"][i]["high_pct"],
            }

        # Invalidate cached merge blend so the next overlap pass reblends with the new images
        merge = self.workflow.data.get("merge", {})
        if "merge_norm_img" in merge:
            self.workflow.data["merge"]["merge_norm_img"] = None

        if not self._pending_predict:
            self._show_busy_message("Updating viewer...")
        self._update_norm_img_layers(ch_indices=list(worker_output.keys()))

        if self._pending_predict:
            self._pending_predict = False
            self._on_img_loaded_and_predict_clicked()
        else:
            self._set_ui_enabled(True)
            self._set_message("Images renormalized", MessageLevel.CHECK)

    def _update_norm_img_layers(self, ch_indices: list | None = None):
        """Update the napari image layers for the given channel indices (default: both)."""

        if ch_indices is None:
            ch_indices = [0, 1]

        with self.viewer.layers.events.blocker():
            for k in ch_indices:
                v = self.workflow.ch_names[k]
                self.layers[k]["image"].data = self.workflow.data[v]["norm_img"]
                self.layers[k]["image"].reset_contrast_limits_range()
                self.layers[k]["image"].reset_contrast_limits()

    def _on_predict_clicked(self):

        # Trigger load worker thread if 'Predict cells' button was clicked directly
        if self.state.workflow_state == WorkflowState.LOAD_IMAGE:

            # Record that prediction has been requested
            self._set_workflow_state(state=WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI)

            self._set_message("Loading images...", MessageLevel.WORK)

            # Trigger load worker thread
            self._start_load_thread()

        elif self.state.workflow_state == WorkflowState.PREDICT_ROI:

            # Renormalize any channel whose clip-pct values have changed before predicting,
            # so prediction always runs on images that match the displayed clip values.
            changed = self._get_changed_pct_channels()
            if changed:
                self._pending_predict = True
                self._set_ui_enabled(False)
                self._start_worker_thread(
                    worker_class=NormalizationWorker,
                    worker_args=(self.workflow.data, self.workflow.ch_names, self.config, changed),
                    success_handler=self._on_normalization_finished,
                    thread_attr_name="_normalization_worker_thread",
                    worker_attr_name="_normalization_worker",
                )
            else:
                self._set_workflow_state(state=WorkflowState.PREDICTING_ROI)
                self._on_img_loaded_and_predict_clicked()

    def _start_load_thread(self):

        self._start_worker_thread(
            worker_class=LoadWorker,
            worker_args=(self.workflow.path, self.config),
            success_handler=self._on_load_finished,
            thread_attr_name="_load_worker_thread",
            worker_attr_name="_load_worker",
        )

    def _on_load_finished(self, worker_output: dict):

        data = worker_output["data"]
        metadata = worker_output["metadata"]

        self.workflow.data = data
        self.workflow.metadata = metadata

        # Warn if channel names are not consistent between metadata and config
        ch0_nm_ok = (
            metadata["channels"][0]["name"] == self.config["channels"][0]["name"]
        )
        ch1_nm_ok = (
            metadata["channels"][1]["name"] == self.config["channels"][1]["name"]
        )

        if not (ch0_nm_ok and ch1_nm_ok):

            self.workflow.channel_mismatch = True
            self.config["channels"][0]["name"] = metadata["channels"][0]["name"]
            self.config["channels"][1]["name"] = metadata["channels"][1]["name"]

        self.workflow.ch_names = {}

        for i in [0, 1]:

            self.workflow.ch_names[i] = metadata["channels"][i]["name"]

        # Update clip-pct box labels to show real channel names
        self._update_clip_pct_box_labels()

        # Record the pct values used for this initial load
        for i in [0, 1]:
            self._last_norm_pcts[i] = {
                "low_pct": self.config["channels"][i]["low_pct"],
                "high_pct": self.config["channels"][i]["high_pct"],
            }

        # Rename "Load images" to "Renormalize images" and rewire its click handler
        if self.ui.btn_load is not None:
            self.ui.btn_load.clicked.disconnect()
            self.ui.btn_load.clicked.connect(self._on_reload_clicked)
            self.ui.btn_load.setText("Renormalize images")

        self._show_busy_message("Updating viewer...")

        self._update_viewer_data_on_load_finished()

        # Stop here if 'Load images' button was clicked
        if self.state.workflow_state == WorkflowState.LOADING_IMAGE:

            self._set_workflow_state(state=WorkflowState.PREDICT_ROI)

            self._set_message(
                "Image loaded",
                MessageLevel.CHECK,
            )

        # Alternatively, proceed with prediction if 'Predict cells' button was clicked
        elif self.state.workflow_state == WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI:

            self._set_workflow_state(state=WorkflowState.IMAGE_LOADED_FOR_PREDICT_ROI)

            # Proceed with prediction
            self._on_img_loaded_and_predict_clicked()

    def _on_img_loaded_and_predict_clicked(self):

        self._set_workflow_state(state=WorkflowState.PREDICTING_ROI)

        self._set_message("Predicting ROIs...", MessageLevel.WORK)

        # Trigger predict worker thread
        self._start_predict_filter_thread(do_predict=True)

    def _start_predict_filter_thread(self, do_predict=True):

        self._start_worker_thread(
            worker_class=PredictFilterWorker,
            worker_args=(
                self.workflow.data,
                self.workflow.metadata,
                self.config,
                self.workflow.ch_names,
                self.models,
                do_predict,
            ),
            success_handler=self._on_predict_filter_finished,
            thread_attr_name="_predict_filter_worker_thread",
            worker_attr_name="_predict_filter_worker",
        )

    def _on_filter_size_clicked(self):

        self._start_predict_filter_thread(do_predict=False)

    def _select_channels(self, title: str, prompt: str):
        """Show the channel-selection dialog; return the chosen channel indices.

        Returns an empty list if the dialog is cancelled or nothing is selected.
        """

        dialog = ChannelSelectWindow(
            title=title,
            channel_names=self.workflow.ch_names,
            prompt=prompt,
            parent=self,
        )

        if not dialog.exec_():
            return []

        return dialog.selected_channels()

    def _on_discard_additions_clicked(self):

        # Ask which channel(s) to clear, then drop the manually drawn ROIs (the
        # shapes layer) for each. StarDist masks (labels layer) are untouched.
        channels = self._select_channels(
            title="Discard added ROIs",
            prompt="Remove all manually added ROIs in:",
        )

        for k in channels:
            shapes = self.layers[k]["shapes"]
            with shapes.events.blocker():
                shapes.data = []

    def _on_discard_deletions_clicked(self):

        # Ask which channel(s) to restore, then reset the labels layer back to the
        # clean size-filtered StarDist prediction (filt_msk), bringing back every
        # manually erased cell. Manually added ROIs (the shapes layer) are untouched.
        channels = self._select_channels(
            title="Restore deleted ROIs",
            prompt="Restore all manually deleted StarDist ROIs in:",
        )

        for k in channels:
            ch_nm = self.workflow.ch_names[k]
            labels = self.layers[k]["labels"]
            with labels.events.blocker():
                # Hand over a copy so subsequent erasing does not mutate the stored
                # filt_msk (see _update_viewer_data_on_predict_filter_finished).
                labels.data = self.workflow.data[ch_nm]["filt_msk"].copy()

    def _on_predict_filter_finished(self, worker_output: dict):

        output = worker_output["output"]
        changed = worker_output["changed"]

        for k, v in output.items():

            self.workflow.data[k].update(v)

        self._show_busy_message("Updating viewer...")

        self._update_viewer_data_on_predict_filter_finished(filt_msk_changed=changed)

        self._set_workflow_state(state=WorkflowState.APPLY_EDITS)

        self._set_message(
            "Prediction finished",
            MessageLevel.CHECK,
        )

    def _rebuild_predict_roi_panel(self, add_change_models: bool = False):
        """Rebuild the PREDICT_ROI widget panel from a blank layout.

        Layout after rebuild:
          0: btn_change_models (created here when add_change_models=True, else already present)
          1-4: clip-pct boxes
          5: Renormalize images button
          6-7: min-area boxes
          8: Predict cells button
          (+ btn_clear already at end)
        """
        if add_change_models:
            btn_change_models = QPushButton("Change models")
            btn_change_models.clicked.connect(self._on_change_models_clicked)
            self.ui.btn_change_models = btn_change_models
            self.layout.insertWidget(0, btn_change_models)

        self._add_clip_pct_boxes(start=1)
        self._update_clip_pct_box_labels()

        btn_load = QPushButton("Renormalize images")
        btn_load.clicked.connect(self._on_reload_clicked)
        self.ui.btn_load = btn_load
        self.layout.insertWidget(5, btn_load)

        self._add_min_area_boxes(insert_pos=6)

        btn_predict = QPushButton("Predict cells")
        btn_predict.clicked.connect(self._on_predict_clicked)
        self.ui.btn_predict = btn_predict
        self.layout.insertWidget(8, btn_predict)

        self._set_viewer_state(ViewerState.IMAGE_LOADED)
        self.state.workflow_state = WorkflowState.PREDICT_ROI

        # btn_clear persists from before model loading and may have been disabled
        # by _set_ui_enabled(False) when actual model loading ran. Re-enable it.
        if self.ui.btn_clear is not None:
            self.ui.btn_clear.setEnabled(True)

        self._set_message("", MessageLevel.NONE)

    def _on_repredict_clicked(self):

        # Tear down the apply-edits panel (btn_repredict + min-area boxes + edit buttons)
        self._destroy_ui_widgets([
            "btn_repredict",
            "box_min_area_ch0",
            "box_min_area_ch1",
            "btn_filter_size",
            "btn_discard_additions",
            "btn_discard_deletions",
            "btn_apply_edits",
        ])

        # btn_change_models is still at index 0 (was not destroyed); rebuild
        # the rest of the PREDICT_ROI panel after it.
        self._rebuild_predict_roi_panel(add_change_models=False)

    def _on_apply_edits_clicked(self):

        # The merge layer is hidden during the compute (APPLYING_EDITS / OVERLAPPING_ROI
        # map to the IMAGE_LOADED viewer state) and only revealed with the status image
        # once overlaps finish, so there is no stale/placeholder merge to manage here.
        self._set_workflow_state(state=WorkflowState.APPLYING_EDITS)

        self._set_message("Applying edits...", MessageLevel.WORK)

        # Trigger apply edits worker thread
        self._start_apply_edits_thread()

    def _start_apply_edits_thread(self):

        copied_masks = {}
        copied_shapes = {}

        for k, v in self.workflow.ch_names.items():

            copied_masks[v] = self.layers[k]["labels"].data.copy()
            copied_shapes[v] = self.layers[k]["shapes"].data

        self._start_worker_thread(
            worker_class=ApplyEditsWorker,
            worker_args=(
                copied_masks,
                copied_shapes,
                self.workflow.data,
                self.workflow.metadata,
                self.config,
                self.workflow.ch_names,
            ),
            success_handler=self._on_apply_edits_finished,
            thread_attr_name="_apply_edits_worker_thread",
            worker_attr_name="_apply_edits_worker",
        )

    def _on_apply_edits_finished(self, worker_output: dict):

        # ApplyEditsWorker only outputs per-channel edit data (the merge image is
        # built by OverlapWorker), so every key is a channel sub-dict.
        for k, v in worker_output.items():

            self.workflow.data[k].update(v)

        # No viewer update here: the merge layer stays hidden through the compute and
        # is revealed with the status image when overlaps finish. Overlap analysis no
        # longer needs a separate "Find overlaps" click: chain straight into the
        # overlap worker and land on the save-results window.
        self._set_workflow_state(state=WorkflowState.OVERLAPPING_ROI)

        self._set_message("Computing overlaps...", MessageLevel.WORK)

        # Trigger overlap worker thread
        self._start_overlap_thread()

    def _on_overlap_filter_clicked(self):

        self._manual_selections.clear()

        for btn in self.ui.btns_choose_rois.values():
            btn.setText("Choose ROIs")

        self._set_message("Re-computing overlaps...", MessageLevel.WORK)

        # Trigger overlap worker thread
        self._start_overlap_thread()

    def _teardown_overlap_filter_or_save_ui(self):

        # Remove the save-results widgets so the panel can be rebuilt for an
        # earlier step (Edit ROIs) or repopulated on the next overlap pass.
        self._destroy_ui_widgets(
            [
                "box_min_pct_ovl_ch0_by_ch1",
                "box_min_pct_ovl_ch1_by_ch0",
                "btn_overlap_filter",
                "btn_edit_rois",
                "btn_save",
            ]
        )

        self._destroy_elem_boxes()

    def _on_edit_rois_clicked(self):

        # Step back from the save-results window to the ROI-editing step so the
        # user can add / remove ROIs without clearing and re-predicting. The
        # editing layers still hold the masks and drawn shapes from the last edit,
        # so editing resumes exactly where it left off; re-applying edits then
        # recomputes overlaps and returns here.
        self._teardown_overlap_filter_or_save_ui()

        # Clear stale overlap visualizations so the layers appear empty if the
        # user makes them visible while editing. Both layers are hidden here
        # (visible=False), so this is just a CPU numpy allocation — no GPU cost.
        self.layers["merge"].data = np.zeros_like(self.layers["merge"].data)
        self.layers["rois"].text = {"constant": ""}
        self.layers["rois"].data = np.zeros((0, 2))

        # Manual ROI selections refer to the overlap result that is about to be
        # recomputed; drop them so the next pass starts from the default top-N.
        self._manual_selections.clear()

        # The size-filter boxes were removed when edits were last applied; rebuild
        # them so the apply-edits panel is complete again.
        self._add_min_area_boxes()

        self._set_workflow_state(state=WorkflowState.APPLY_EDITS, force=True)

        # Clear the "Results saved" message left over from the save-results window.
        self._set_message("", MessageLevel.NONE)

    def _start_overlap_thread(self):

        self._start_worker_thread(
            worker_class=OverlapWorker,
            worker_args=(self.workflow.data, self.config, self.workflow.ch_names),
            success_handler=self._on_overlap_finished,
            thread_attr_name="_overlap_worker_thread",
            worker_attr_name="_overlap_worker",
        )

    def _on_overlap_finished(self, worker_output: dict):

        # "merge" and "roi_graphics" are complete dicts built fresh by OverlapWorker
        # (the merge dict may not exist yet on the first pass), so replace them; the
        # remaining keys are per-channel sub-dicts to merge in.
        for k, v in worker_output.items():

            if k in ("roi_graphics", "merge"):

                self.workflow.data[k] = v

            else:

                self.workflow.data[k].update(v)

        # Delay viewer update
        self._show_busy_message("Updating viewer...")

        # Establish the LOCKED viewer state BEFORE applying the overlap data. The
        # LOCKED baseline hides the rois layer; _update_viewer_data_on_overlap_finished
        # then re-shows it on top (and sizes its text for the current zoom). Doing it
        # in the reverse order would re-hide the freshly shown labels — and a later
        # manual toggle would draw them at a stale, oversized text size because the
        # zoom-tracking rescale is skipped while the layer is hidden.
        self._set_viewer_state(ViewerState.LOCKED)

        self._update_viewer_data_on_overlap_finished()

        if self.state.workflow_state == WorkflowState.OVERLAPPING_ROI:

            self._set_workflow_state(state=WorkflowState.OVERLAP_FILTER_OR_SAVE)

        else:

            self._set_workflow_state(
                state=WorkflowState.UPDATE_OVERLAP_FILTER_OR_SAVE, force=True
            )

        self._refresh_selection_summary()

    def _build_export_data(self) -> dict:
        """Shallow subset + reorder of self.workflow.data for pickling.

        Values are shared by reference; self.workflow.data is left untouched so a
        re-save after a parameter change still works. Insertion order == export
        order; a missing key raises KeyError so pipeline changes surface loudly.
        """
        data = self.workflow.data

        def pick(d: dict, keys: tuple[str, ...]) -> dict:
            return {k: d[k] for k in keys}

        ch_keys = (
            "img",
            "norm_img",
            "msk",
            "filt_msk",
            "edit_msk",
            "submsks_area",
            "submsks_area_um2",
            "cnts",
            "roi_status_data",
            "collection_summary",
        )

        ch0_nm = self.workflow.ch_names[0]
        ch1_nm = self.workflow.ch_names[1]
        out = {
            ch0_nm: pick(data[ch0_nm], ch_keys),
            ch1_nm: pick(data[ch1_nm], ch_keys),
            "merge": data["merge"],
            "roi_id_map": data["roi_id_map"],
            "roi_graphics": data["roi_graphics"],
        }
        return out

    def _on_save_clicked(self):

        # Read content of 'element' boxes
        self._read_elem_boxes()

        # Define output directory path
        out_dir_path = Path(
            Path(self.config["out_dir_path"]).expanduser(),
            self.workflow.metadata["img_nm"],
        )

        # Resolve per-ROI element metadata + old-ID -> new-ID correspondence, then
        # format the element lists for writing.
        elem_records, roi_id_map = workflow.make_elem_metadata(
            data_dict=self.workflow.data,
            metadata_dict=self.workflow.metadata,
            config_dict=self.config,
            summary_key="collection_summary",
            cnts_key="cnts",
            submsks_area_um2_key="submsks_area_um2",
            pop_order=tuple(self.config["elements"].keys()),
            selected_ids_override=(
                self._manual_selections if self._manual_selections else None
            ),
        )

        # Abort cleanly when no ROI is selected for collection: an element list
        # with no elements cannot be formatted (pandas: "No objects to
        # concatenate") and would not be useful to import into PALMRobo.
        if not any(r.collected for r in elem_records):

            self._set_message(
                "The element list is empty: no ROI selected for collection.\n\n"
                "No results were written to file.",
                MessageLevel.WARNING,
            )

            print(
                f"Warning - {self.workflow.metadata['img_nm']}:\n"
                "The element list is empty: no ROI selected for collection.\n"
                "No results were written to file.\n"
            )

            return

        elem_list = workflow.make_elem_list(elem_records, self.workflow.metadata)

        for k, txt in elem_list.items():

            with open(Path(out_dir_path, f"elem_list_{k}.txt"), "w") as f:
                f.write(txt)

        # Record the old-ID -> new-element-ID / collect-omit correspondence so it
        # is persisted alongside the data.
        self.workflow.data["roi_id_map"] = roi_id_map

        # Write data and metadata to file
        workflow.pickle_data(
            data=self._build_export_data(), filename=Path(out_dir_path, "data.pkl")
        )

        workflow.pickle_data(
            data=self.workflow.metadata,
            filename=Path(out_dir_path, "metadata.pkl"),
        )

        # Write config to file
        with open(Path(out_dir_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

        self._set_message(
            f"Results saved at:\n{self.config['out_dir_path']}\n\nIn subfolder:\n{self.workflow.metadata['img_nm']}",
            MessageLevel.SAVE,
        )

        QMessageBox.information(
            self,
            "Results saved",
            f"Results saved at:\n{out_dir_path}",
        )

        # Print the collect/omit summary to the terminal, but only when its content
        # differs from the last one printed, so repeated saves of an unchanged
        # selection don't spam the terminal.
        summary_lines = [
            f"Collect / omit summary - {self.workflow.metadata['img_nm']}:\n"
        ]
        for p in self._populations_in_config_order():
            collection_summary = self.workflow.data[
                self.workflow.ch_names[p.primary_ch]
            ]["collection_summary"]
            summary_lines.append(
                f"- {self._elem_label(p)}: "
                f"{collection_summary[f'{p.status}_collect']} / {collection_summary[p.status]} "
                f"({collection_summary[f'{p.status}_tube_id']})"
            )

        summary = "\n".join(summary_lines)

        with open(Path(out_dir_path, "collection_summary.txt"), "w") as f:
            f.write(summary)

        if summary != self._last_collection_summary_print:
            print(summary)
            print("\n")
            self._last_collection_summary_print = summary

    def _populations_in_config_order(self) -> list[Population]:
        """Populations ordered as the user listed them in config['elements']."""
        return [
            POPULATION_BY_KEY[k]
            for k in self.config["elements"].keys()
            if k in POPULATION_BY_KEY
        ]

    def _elem_label(self, p: Population) -> str:
        """Build a population's display label, e.g. 'TH⁺ / pSyn⁻' or 'TH⁺ / pSyn-amb'."""
        plus = "⁺"
        minus = "⁻"
        suffix = {"neg": minus, "pos": plus, "amb": "-amb"}[p.status]
        primary = self.workflow.ch_names[p.primary_ch]
        secondary = self.workflow.ch_names[1 - p.primary_ch]
        return f"{primary}{plus} / {secondary}{suffix}"

    def _get_elem_box(self, pop_key: str) -> ElementConfigBox:
        # Only called once the element boxes have been built (OVERLAP_FILTER_OR_SAVE).
        box = self.ui.box_elems.get(pop_key)
        assert box is not None
        return box

    def _on_n_collect_changed(self, pop_key: str):
        # Drops any manual selection when the spinner is edited directly. The
        # summary is refreshed by on_elem_params_changed, which fires right
        # after this on the same spin_n.valueChanged signal.
        if pop_key in self._manual_selections:
            self._manual_selections.pop(pop_key)
            if pop_key in self.ui.btns_choose_rois:
                self.ui.btns_choose_rois[pop_key].setText("Choose ROIs")

    def _on_choose_rois_clicked(self, pop_key: str):

        p = POPULATION_BY_KEY[pop_key]
        status = p.status
        ch_nm = self.workflow.ch_names[p.primary_ch]
        ch_other_nm = self.workflow.ch_names[1 - p.primary_ch]

        cnts = self.workflow.data[ch_nm]["cnts"][status]
        status_dict = self.workflow.data[ch_nm]["roi_status_data"][status]

        rois = [(roi_id, status_dict.get(roi_id, {})) for roi_id in cnts.keys()]

        if pop_key in self._manual_selections:
            initial_selected = self._manual_selections[pop_key]
        else:
            box = self._get_elem_box(pop_key)
            n_collect = box.spin_n.value()
            initial_selected = set(list(cnts.keys())[:n_collect])

        px_area_um2 = self.workflow.metadata["image"]["px_area_um2"]
        title = self._get_elem_box(pop_key).base_label

        # Pick which overlap metric to show per cell. The summary keys are
        # perspective-relative to the population's primary channel:
        # "max_pct_ovl_1by2" is always the fraction of the primary cell (ch_nm)
        # overlapped by the secondary cell (ch_other_nm).
        if status == "pos":
            ovl_key = "max_mean_pct_ovl"
            ovl_label = "mean % overlap"
        elif status == "amb":
            ovl_key = "max_pct_ovl_1by2"
            ovl_label = f"% overlap {ch_nm} / {ch_other_nm}"
        else:  # "neg": no overlap summary exists for these cells
            ovl_key = None
            ovl_label = ""

        dialog = ChooseROIsWindow(
            title=title,
            rois=rois,
            ovl_key=ovl_key,
            ovl_label=ovl_label,
            initial_selected=initial_selected,
            px_area_um2=px_area_um2,
            parent=self,
        )

        if dialog.exec_():
            selected = dialog.selected_ids()

            # Treat an unchanged selection like a cancel: leave state and the
            # "Results saved" message untouched.
            if selected == initial_selected:
                return

            self._manual_selections[pop_key] = selected

            box = self._get_elem_box(pop_key)
            box.spin_n.blockSignals(True)
            box.spin_n.setValue(len(selected))
            box.spin_n.blockSignals(False)

            self._refresh_selection_summary()

    def _pop_selection_summary(self, pop_key: str):
        """Return (n_selected, summed_area_um2) for the current selection of pop_key.

        Mirrors the selection logic of the "Choose ROIs" dialog: a manual
        selection if one exists, otherwise the top-N ranked cells per the
        element box's 'n collect' value.
        """
        p = POPULATION_BY_KEY[pop_key]
        status = p.status
        ch_nm = self.workflow.ch_names[p.primary_ch]

        cnts = self.workflow.data[ch_nm]["cnts"][status]
        status_dict = self.workflow.data[ch_nm]["roi_status_data"][status]

        if pop_key in self._manual_selections:
            selected = self._manual_selections[pop_key]
        else:
            n_collect = self._get_elem_box(pop_key).spin_n.value()
            selected = set(list(cnts.keys())[:n_collect])

        px_area_um2 = self.workflow.metadata["image"]["px_area_um2"]
        total_area = sum(
            status_dict.get(cell_id, {}).get("area", 0) * px_area_um2
            for cell_id in selected
        )

        return len(selected), total_area

    def _selection_summary_message(self):
        """Build a message summarising selected ROI count and area per population.

        Any tube-ID matching warning is shown below the summary (and above the
        channel-mismatch warning that _set_message appends afterwards).
        """
        lines = ["Selection summary:\n"]
        for p in self._populations_in_config_order():
            n, area = self._pop_selection_summary(p.key)
            base_label = self._get_elem_box(p.key).base_label
            lines.append(f"- {base_label}: {n} ROIs  |  {area:.0f} µm²")

        message = "\n".join(lines)

        if self._tube_match_warning is not None:
            message += f"\n\n⚠ {self._tube_match_warning}"

        return message

    def _refresh_selection_summary(self):
        """Re-render the per-population selection summary in the message box."""
        self._set_message(self._selection_summary_message(), MessageLevel.CHECK)

    def _on_clip_pct_changed(self, ch_idx: int, key: str, value: float):

        self.config["channels"][ch_idx][key] = float(value)

    def _on_min_area_changed(self, key: int, value: float):

        self.config["channels"][key]["min_area_um2"] = float(value)

    def _on_min_pct_ovl_changed(self, key: str, value: float):

        self.config[key] = float(value)

    def _on_elem_params_changed(self, key: str, values: dict):

        self.config["elements"][key].update(values)

        self._refresh_selection_summary()

    def _resolve_tube_ids(self) -> "tuple[dict[str, str], str | None]":
        """Resolve the tube ID to pre-select for each population.

        Returns a {population key: tube_id} map plus an optional warning message.
        When tube-ID filename matching is disabled, every population maps to its
        config['elements'] default. When enabled, the first set whose 'match'
        string is found (case-insensitive substring) in the loaded file name
        overrides those defaults; if no set matches, the defaults are used and a
        warning is returned. A warning is also returned (first set wins) when the
        file name matches more than one set.
        """

        elements = self.config["elements"]
        defaults = {p.key: elements[p.key]["tube_id"] for p in POPULATIONS}

        matching = elements["tube_id_matching"]

        if not matching["enabled"]:
            return defaults, None

        stem = Path(self.workflow.path).stem
        stem_cf = stem.casefold()

        matched = [s for s in matching["sets"] if s["match"].casefold() in stem_cf]

        if not matched:
            return defaults, (
                f"No tube-ID set matched file name '{stem}'; using default tubes."
            )

        chosen = matched[0]

        warning = None
        if len(matched) > 1:
            names = ", ".join(f"'{s['match']}'" for s in matched)
            warning = (
                f"File name '{stem}' matched multiple tube-ID sets ({names}); "
                f"using the first ('{chosen['match']}')."
            )

        resolved = {
            p.key: chosen["tube_ids"].get(p.key, defaults[p.key]) for p in POPULATIONS
        }

        return resolved, warning

    def _update_elem_boxes(self):

        tube_ids, self._tube_match_warning = self._resolve_tube_ids()

        for p in POPULATIONS:

            box = self.ui.box_elems.get(p.key)

            if box is None:
                continue

            collection_summary = self.workflow.data[
                self.workflow.ch_names[p.primary_ch]
            ]["collection_summary"]
            elem_cfg = self.config["elements"][p.key]

            max_n_collect = collection_summary[p.status]
            n_collect = max_n_collect if elem_cfg["collect"] is True else 0

            box.label.setText(f"{box.base_label} ({max_n_collect} total)")
            box.label.setStyleSheet(f"font-weight: bold; color: {elem_cfg['color']}")

            box.spin_n.blockSignals(True)
            box.spin_n.setMaximum(max_n_collect)
            box.spin_n.setValue(n_collect)
            box.spin_n.blockSignals(False)

            box.combo_laser.blockSignals(True)
            box.combo_laser.setCurrentText(elem_cfg["laser_function"])
            box.combo_laser.blockSignals(False)

            box.combo_tube.blockSignals(True)
            box.combo_tube.setCurrentText(tube_ids[p.key])
            box.combo_tube.blockSignals(False)

    def _read_elem_boxes(self):

        for p in POPULATIONS:

            box = self.ui.box_elems.get(p.key)
            assert box is not None

            collection_summary = self.workflow.data[
                self.workflow.ch_names[p.primary_ch]
            ]["collection_summary"]

            collection_summary[f"{p.status}_collect"] = box.spin_n.value()
            collection_summary[f"{p.status}_laser_function"] = (
                box.combo_laser.currentText()
            )
            collection_summary[f"{p.status}_tube_id"] = box.combo_tube.currentText()

        self.workflow.data[self.workflow.ch_names[0]]["collection_summary"] = {
            k: self.workflow.data[self.workflow.ch_names[0]]["collection_summary"][k]
            for k in [
                "total",
                "neg",
                "pos",
                "amb",
                "neg_collect",
                "pos_collect",
                "amb_collect",
                "neg_laser_function",
                "pos_laser_function",
                "amb_laser_function",
                "neg_tube_id",
                "pos_tube_id",
                "amb_tube_id",
            ]
        }

        self.workflow.data[self.workflow.ch_names[1]]["collection_summary"] = {
            k: self.workflow.data[self.workflow.ch_names[1]]["collection_summary"][k]
            for k in [
                "total",
                "neg",
                "pos",
                "amb",
                "neg_collect",
                "amb_collect",
                "neg_laser_function",
                "amb_laser_function",
                "neg_tube_id",
                "amb_tube_id",
            ]
        }
