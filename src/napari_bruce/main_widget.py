# %% Set up ----

# Load dependencies
import sys
import os
import copy
import cv2
import json
import importlib.resources
import traceback
import numpy as np
from pathlib import Path
from enum import Enum, auto
from matplotlib.colors import to_rgba
from contextlib import redirect_stdout, redirect_stderr
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
    QCheckBox,
    QDialogButtonBox,
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

# %% WorkflowState() ----


class WorkflowState(Enum):

    SELECT_FILE = auto()
    LOAD_IMAGE = auto()
    LOADING_IMAGE = auto()
    LOADING_IMAGE_FOR_PREDICT_ROI = auto()
    PREDICT_ROI = auto()
    IMAGE_LOADED_FOR_PREDICT_ROI = auto()
    PREDICTING_ROI = auto()
    APPLY_EDITS = auto()
    APPLYING_EDITS = auto()
    OVERLAP_ROI = auto()
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

    btn_select: QPushButton | None
    btn_clear: QPushButton | None
    btn_load: QPushButton | None
    btn_predict: QPushButton | None
    btn_filter_size: QPushButton | None
    btn_apply_edits: QPushButton | None
    btn_overlap: QPushButton | None
    btn_overlap_filter: QPushButton | None
    btn_save: QPushButton | None
    box_min_area_ch0: "ParamValueBox | None"
    box_min_area_ch1: "ParamValueBox | None"
    box_min_pct_ovl_ch0_by_ch1: "ParamValueBox | None"
    box_min_pct_ovl_ch1_by_ch0: "ParamValueBox | None"
    box_elems: "dict[str, ElementConfigBox]"
    btns_choose_rois: dict[str, QPushButton]

    def __init__(self):

        self.reset()

    def reset(self):

        self.btn_select = None
        self.btn_clear = None
        self.btn_load = None
        self.btn_predict = None
        self.btn_filter_size = None
        self.btn_apply_edits = None
        self.btn_overlap = None
        self.btn_overlap_filter = None
        self.btn_save = None
        self.box_min_area_ch0 = None
        self.box_min_area_ch1 = None
        self.box_min_pct_ovl_ch0_by_ch1 = None
        self.box_min_pct_ovl_ch1_by_ch0 = None
        self.box_elems = {}
        self.btns_choose_rois = {}


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

    # Methods
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

    def __init__(self, title, rois, initial_selected, px_area_um2=1.0, parent=None):
        """
        rois: ordered list of (ROI ID, roi_data_dict) — best ranked first.
        initial_selected: set of ROI IDs to pre-check.
        roi_data_dict has 'area' (pixels) and optionally 'summary' with overlap metrics.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(340, 460)

        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_none = QPushButton("Select none")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

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
            if "summary" in roi_data:
                ovl = roi_data["summary"].get("max_mean_pct_ovl", 0)
                label += f"  |  {ovl:.1f}% ovl"
            roi_box = QCheckBox(label)
            roi_box.setChecked(roi_id in initial_selected)
            roi_box.stateChanged.connect(self._update_total_area)
            self._checkboxes[roi_id] = roi_box
            scroll_layout.addWidget(roi_box)

        scroll_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        self._total_area_label = QLabel()
        self._total_area_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._total_area_label)
        self._update_total_area()

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


# %% LoadModelWorker() ----


class LoadModelWorker(BaseWorker):

    def __init__(self, config: dict, parent=None):

        super().__init__(parent)
        self.config = config

    def compute(self):

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):

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

            if self.do_predict:

                # Run StarDist
                img = self.data[v]["norm_img"].astype(np.float32) / 255.0

                n_tiles = workflow.choose_stardist_n_tiles(img)

                with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):

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

            ch_submsks_area = dict(self.data[v]["submsks_area"])
            ch_submsks_area_um2 = dict(self.data[v]["submsks_area_um2"])

            # If user drew cell shapes, add them to the edited cell predictions and compute their area
            if len(ch_shapes) > 0:

                start_idx = int(np.max(ch_msk) + 1)

                ch_msk = workflow.append_shapes_to_msk(
                    msk=ch_msk, shapes=ch_shapes, start_idx=start_idx
                )

                ch_msk = ch_msk.astype(np.uint16)

                new_ch_submsks_area = workflow.count_submsks_pixels(
                    msk=np.where(ch_msk >= start_idx, ch_msk, 0)
                )

                for i, j in new_ch_submsks_area.items():

                    if i not in ch_submsks_area.keys():

                        ch_submsks_area[i] = j

                        ch_submsks_area_um2[i] = j * px_area_um2

            ch_cnt = workflow.msk_to_cnts(msk=ch_msk)

            output[v] = {
                "edit_msk": ch_msk,
                "edit_cnts": ch_cnt,
                "submsks_area": ch_submsks_area,
                "submsks_area_um2": ch_submsks_area_um2,
            }

        # Produce base merge image
        ch0 = self.data[self.ch_names[0]]["norm_img"].astype(np.float32)
        ch1 = self.data[self.ch_names[1]]["norm_img"].astype(np.float32)

        c0 = to_rgba(self.config["channels"][0]["color"])
        c1 = to_rgba(self.config["channels"][1]["color"])

        scale = 0.8
        merge_r = scale * (ch0 * c0[0] + ch1 * c1[0])
        merge_g = scale * (ch0 * c0[1] + ch1 * c1[1])
        merge_b = scale * (ch0 * c0[2] + ch1 * c1[2])

        merge_norm_img = np.stack([merge_r, merge_g, merge_b], axis=-1)
        merge_norm_img = np.clip(merge_norm_img, 0, 255).astype(np.uint8)

        merge_norm_img_rois = merge_norm_img.copy()

        # Add ch0 / ch1 ROIs to merge image
        for i, j in zip(
            [self.ch_names[0], self.ch_names[1]],
            [tuple(int(x * 255) for x in c0[:3]), tuple(int(x * 255) for x in c1[:3])],
        ):

            merge_norm_img_rois = cv2.drawContours(
                image=merge_norm_img_rois,
                contours=[x for x in output[i]["edit_cnts"].values()],
                contourIdx=-1,
                color=j,
                thickness=4,
            )

        output["merge"] = {
            "merge_norm_img": merge_norm_img,
            "merge_norm_img_rois": merge_norm_img_rois,
        }

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

        # Compute cell status
        status_ch0, status_ch1 = workflow.get_submsks1_submsks2_status(
            msk1=self.data[ch0_nm]["edit_msk"],
            msk2=self.data[ch1_nm]["edit_msk"],
            min_pct_ovl_1by2=self.config["min_pct_ovl_ch0_by_ch1"],
            min_pct_ovl_2by1=self.config["min_pct_ovl_ch1_by_ch0"],
            submsks1_pix_dict=self.data[ch0_nm]["submsks_area"],
            submsks2_pix_dict=self.data[ch1_nm]["submsks_area"],
        )

        # For each channel...
        for i, j, k in [(ch0_nm, ch1_nm, status_ch0), (ch1_nm, ch0_nm, status_ch1)]:

            # Produce cell status summary
            summary = {x: len(y) for x, y in k.items()}
            summary["total"] = sum(summary.values())

            # Extract contours per status from the precomputed contour dict
            cnts = workflow.status_dict_to_cnts(
                status_dict=k,
                cnt_dict=self.data[i]["edit_cnts"],
            )

            output[i] = {
                f"{j}_status": k,
                "summary": summary,
                f"cnts": cnts,
            }

        merge_norm_img = self.data["merge"]["merge_norm_img"].copy()

        def rgb_from_config(key):

            return tuple(
                int(x * 255) for x in to_rgba(self.config["elements"][key]["color"])[:3]
            )

        ch_nm_by_idx = (ch0_nm, ch1_nm)

        for p in POPULATIONS:

            merge_norm_img = cv2.drawContours(
                image=merge_norm_img,
                contours=[
                    x
                    for x in output[ch_nm_by_idx[p.primary_ch]]["cnts"][
                        p.status
                    ].values()
                ],
                contourIdx=-1,
                color=rgb_from_config(p.key),
                thickness=4,
            )

        # Collect centroids, rank labels and population colors for the ROI IDs Points layer
        def _centroid_yx(cnt):
            M = cv2.moments(cnt.reshape(-1, 1, 2).astype(np.int32))
            if M["m00"] == 0:
                return None
            return int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"])  # (row, col)

        ranked_ids, centroids, colors = [], [], []

        for p in POPULATIONS:
            ch_nm = ch_nm_by_idx[p.primary_ch]
            rgba = tuple(c / 255.0 for c in rgb_from_config(p.key)) + (1.0,)
            for rank, cnt in enumerate(
                output[ch_nm]["cnts"][p.status].values(), start=1
            ):
                pt = _centroid_yx(cnt)
                if pt is not None:
                    ranked_ids.append(str(rank))
                    centroids.append(pt)
                    colors.append(rgba)

        output["rois"] = {
            "ranked_ids": ranked_ids,
            "centroids": (
                np.array(centroids, dtype=float) if centroids else np.zeros((0, 2))
            ),
            "colors": (np.array(colors, dtype=float) if colors else np.zeros((0, 4))),
        }

        output["merge"] = {"merge_norm_img_status": merge_norm_img}

        return output


# %% _make_color_overlay_delegate() ----


def _make_color_overlay_delegate(base_cls):
    class _ColorOverlayDelegate(base_cls):
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

    return _ColorOverlayDelegate


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

        # ROI IDs text size is in data coordinates, so it scales with zoom; these track
        # a reference zoom to rescale the font and keep its on-screen size constant.
        self._roi_id_base_text_size = 10
        self._roi_id_ref_zoom = None

        self._build_layout()

        self._init_viewer_layers()

        self._install_layer_colors()

        self.viewer.camera.events.zoom.connect(self._update_roi_id_text_size)

        message = f"""Loading StarDist models...
    
    Channel 0: {self.config['channels'][0]['stardist_model']}
    Channel 1: {self.config['channels'][1]['stardist_model']}
    """

        self._set_message(message, MessageLevel.WORK)

        self._start_worker_thread(
            worker_class=LoadModelWorker,
            worker_args=(self.config,),
            success_handler=self._on_models_loaded,
            thread_attr_name="_load_model_worker_thread",
            worker_attr_name="_load_model_worker",
        )

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

    def closeEvent(self, event):

        self._cleanup_all_workers()

        super().closeEvent(event)

    def _cleanup_all_workers(self):

        thread_attr_names = [
            "_load_model_worker_thread",
            "_load_worker_thread",
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

        worker.sig_error.connect(
            lambda msg, worker=worker_class.__name__: self._on_worker_error(worker, msg)
        )

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

        self._set_message(
            f"{worker_name} failed with the following error:\n\n{error_msg}",
            MessageLevel.ERROR,
        )

        assert self.ui.btn_clear is not None
        self.ui.btn_clear.setEnabled(True)

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

        self.setLayout(self.layout)
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
        }:

            viewer_state = ViewerState.IMAGE_LOADED

        elif workflow_state in {WorkflowState.APPLY_EDITS}:

            viewer_state = ViewerState.ROI_EDITING

        elif workflow_state in {
            WorkflowState.APPLYING_EDITS,
            WorkflowState.OVERLAP_ROI,
            WorkflowState.OVERLAPPING_ROI,
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

        if workflow_state == WorkflowState.SELECT_FILE:

            self._set_select_file_ui()

        elif workflow_state == WorkflowState.LOAD_IMAGE:

            self._set_load_image_ui()

        elif workflow_state in {
            WorkflowState.LOADING_IMAGE,
            WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI,
        }:

            self._set_loading_image_ui()

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

        elif workflow_state == WorkflowState.OVERLAP_ROI:

            self._set_overlap_roi_ui()

        elif workflow_state == WorkflowState.OVERLAPPING_ROI:

            self._set_overlapping_roi_ui()

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

    def _set_select_file_ui(self):

        btn_select = QPushButton("Select file")
        btn_select.clicked.connect(self._on_select_clicked)
        self.header_layout.insertWidget(0, btn_select)
        self.ui.btn_select = btn_select

    def _set_load_image_ui(self):

        # Remove 'Select file' button from viewer
        btn_select = self.ui.btn_select
        assert btn_select is not None
        self.layout.removeWidget(btn_select)
        btn_select.hide()
        btn_select.deleteLater()
        self.ui.btn_select = None

        # Add 'Load images', 'Predict cells' and 'Clear' buttons to viewer
        btn_load = QPushButton("Load images")
        btn_load.clicked.connect(self._on_load_clicked)
        self.ui.btn_load = btn_load

        btn_predict = QPushButton("Predict cells")
        btn_predict.clicked.connect(self._on_predict_clicked)
        self.ui.btn_predict = btn_predict

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda checked=False: self._on_clear_clicked())
        self.ui.btn_clear = btn_clear

        for i, j in zip([0, 1, 2], [btn_load, btn_predict, btn_clear]):

            self.layout.insertWidget(i, j)

    def _set_loading_image_ui(self):

        # Remove 'Load images' button from viewer
        btn_load = self.ui.btn_load
        assert btn_load is not None
        self.layout.removeWidget(btn_load)
        btn_load.hide()
        btn_load.deleteLater()
        self.ui.btn_load = None

    def _set_predict_roi_ui(self, loaded_for_predict=False):

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

        for i, j in zip([0, 1], [box_min_area_ch0, box_min_area_ch1]):

            self.layout.insertWidget(i, j)

        if loaded_for_predict:

            # Disable 'min n pix' boxes
            for i in [box_min_area_ch0, box_min_area_ch1]:

                i.setEnabled(False)

        else:

            # Re-enable 'Predict cells' and 'Clear' buttons
            for i in [self.ui.btn_predict, self.ui.btn_clear]:

                assert i is not None
                i.setEnabled(True)

    def _set_predicting_roi_ui(self):

        for i in ["btn_load", "btn_predict"]:

            j = getattr(self.ui, i, None)

            if j is not None:

                self.layout.removeWidget(j)
                j.hide()
                j.deleteLater()
                setattr(self.ui, i, None)

    def _set_apply_edits_ui(self):

        # Add 'Adjust size filter' and 'Apply edits' buttons to viewer if predictions are returned for the first time
        btn_filter_size = QPushButton("Adjust size filter")
        btn_filter_size.clicked.connect(self._on_filter_size_clicked)
        self.ui.btn_filter_size = btn_filter_size

        btn_apply_edits = QPushButton("Apply edits")
        btn_apply_edits.clicked.connect(self._on_apply_edits_clicked)
        self.ui.btn_apply_edits = btn_apply_edits

        for i, j in zip([2, 3], [btn_filter_size, btn_apply_edits]):

            self.layout.insertWidget(i, j)

        # Re-enable 'min n pix' boxes, 'Adjust size filter', 'Apply edits' and 'Clear' buttons
        # => 'Adjust size filter' and 'Apply edits' buttons are already enabled if predictions are returned for the first time
        for i in [
            self.ui.box_min_area_ch0,
            self.ui.box_min_area_ch1,
            self.ui.btn_filter_size,
            self.ui.btn_apply_edits,
            self.ui.btn_clear,
        ]:

            assert i is not None
            i.setEnabled(True)

    def _set_applying_edits_ui(self):

        # Remove 'min n pix' boxes, 'Adjust size filter' and 'Apply edits' buttons from viewer
        for i in [
            "box_min_area_ch0",
            "box_min_area_ch1",
            "btn_filter_size",
            "btn_apply_edits",
        ]:

            j = getattr(self.ui, i, None)

            if j is not None:

                self.layout.removeWidget(j)
                j.hide()
                j.deleteLater()
                setattr(self.ui, i, None)

    def _set_overlap_roi_ui(self):

        # Add 'min % ovl' boxes and 'Find overlaps' button to viewer
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

        btn_overlap = QPushButton("Find overlaps")
        btn_overlap.clicked.connect(self._on_overlap_clicked)
        self.ui.btn_overlap = btn_overlap

        for i, j in zip(
            [0, 1, 2],
            [
                box_min_pct_ovl_ch0_by_ch1,
                box_min_pct_ovl_ch1_by_ch0,
                btn_overlap,
            ],
        ):

            self.layout.insertWidget(i, j)

        # Re-enable 'Clear' button
        assert self.ui.btn_clear is not None
        self.ui.btn_clear.setEnabled(True)

    def _set_overlapping_roi_ui(self):

        # Remove 'Find overlaps' button from viewer
        btn_overlap = self.ui.btn_overlap
        assert btn_overlap is not None
        self.layout.removeWidget(btn_overlap)
        btn_overlap.hide()
        btn_overlap.deleteLater()
        self.ui.btn_overlap = None

    def _set_overlap_filter_or_save_ui(self, build_ui=True):

        if build_ui:

            # Add 'Adjust overlap filter' button, 'element' boxes and 'Save results' button to viewer if overlaps are returned for the first time
            btn_overlap_filter = QPushButton("Adjust overlap filter")
            btn_overlap_filter.clicked.connect(self._on_overlap_filter_clicked)
            self.ui.btn_overlap_filter = btn_overlap_filter

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

            # Insert widgets: btn_overlap_filter, element boxes, btn_save
            widgets_to_insert = [btn_overlap_filter] + box_elem_list + [btn_save]

            for i, w in enumerate(widgets_to_insert):
                self.layout.insertWidget(2 + i, w)

        # Re-enable 'min % ovl' and 'element' boxes, 'Adjust overlap filter', 'Save results' and 'Clear' buttons
        # => 'Adjust overlap filter' button, 'element' boxes and 'Save results' button do not exist yet if overlaps are returned for the first time
        for i in [
            self.ui.box_min_pct_ovl_ch0_by_ch1,
            self.ui.box_min_pct_ovl_ch1_by_ch0,
            self.ui.btn_overlap_filter,
            *self.ui.box_elems.values(),
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

        for i in [
            "btn_clear",
            "btn_load",
            "btn_predict",
            "btn_filter_size",
            "box_min_area_ch0",
            "box_min_area_ch1",
            "btn_apply_edits",
            "btn_overlap",
            "btn_overlap_filter",
            "box_min_pct_ovl_ch0_by_ch1",
            "box_min_pct_ovl_ch1_by_ch0",
            "btn_save",
        ]:

            j = getattr(self.ui, i, None)

            if j is not None:

                self.layout.removeWidget(j)
                j.hide()
                j.deleteLater()
                setattr(self.ui, i, None)

        # Tear down the per-population element boxes (held in a dict, not named attrs).
        for box in self.ui.box_elems.values():
            self.layout.removeWidget(box)
            box.hide()
            box.deleteLater()
        self.ui.box_elems.clear()

        # The "Choose ROIs" buttons live inside the element boxes and are destroyed
        # together with them above; just drop the references.
        self.ui.btns_choose_rois.clear()

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

    def _update_viewer_data_on_apply_edits_finished(self):

        with self.viewer.layers.events.blocker():

            self.layers["merge"].data = self.workflow.data["merge"][
                "merge_norm_img_rois"
            ]

    def _update_viewer_data_on_overlap_finished(self):

        rois = self.workflow.data.get("rois", {})
        ranked_ids = rois.get("ranked_ids", [])
        centroids = rois.get("centroids", np.zeros((0, 2)))
        colors = rois.get("colors", np.zeros((0, 4)))

        self.layers["merge"].data = self.workflow.data["merge"]["merge_norm_img_status"]

        pts = centroids if len(centroids) > 0 else np.zeros((0, 2))
        layer = self.layers["rois"]

        # Capture the current zoom as the reference at which base text size applies.
        self._roi_id_ref_zoom = self.viewer.camera.zoom

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
            # Set text before AND after data: assigning .data resets the TextManager,
            # so the labels must be reapplied afterwards to survive the reset.
            layer.text = text_dict
            layer.data = pts
            layer.text = text_dict
            layer.visible = True
        else:
            layer.data = pts
            layer.visible = True

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
                f"{text}\n\n\nWARNING: channel name mismatch between image and config."
            )

        self.msg_text.setText(text)

        if level.value is None:

            self.msg_icon.setVisible(False)

        else:

            icon = QApplication.style().standardIcon(level.value)
            self.msg_icon.setPixmap(icon.pixmap(16, 16))
            self.msg_icon.setVisible(True)

    def _on_models_loaded(self, models: dict):

        self.models = models

        self._set_message("", MessageLevel.NONE)

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

        self._set_workflow_state(state=WorkflowState.CLEARED)

        self.workflow.reset()

        self.ui.reset()

        self.state.reset()

        self._reset_viewer_layers()

        self._set_workflow_state(state=WorkflowState.SELECT_FILE)

        self.file_label.setText("")
        self.file_label.setVisible(False)

        self._set_message("", MessageLevel.NONE)

    def _on_load_clicked(self):

        self._set_workflow_state(state=WorkflowState.LOADING_IMAGE)

        self._set_message("Loading images...", MessageLevel.WORK)

        # Trigger load worker thread
        self._start_load_thread()

    def _on_predict_clicked(self):

        # Trigger load worker thread if 'Predict cells' button was clicked directly
        if self.state.workflow_state == WorkflowState.LOAD_IMAGE:

            # Record that prediction has been requested
            self._set_workflow_state(state=WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI)

            self._set_message("Loading images...", MessageLevel.WORK)

            # Trigger load worker thread
            self._start_load_thread()

        elif self.state.workflow_state == WorkflowState.PREDICT_ROI:

            # Alternatively, proceed with prediction if 'Load images' button was clicked previously
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

        self._set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

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

    def _on_predict_filter_finished(self, worker_output: dict):

        output = worker_output["output"]
        changed = worker_output["changed"]

        for k, v in output.items():

            self.workflow.data[k].update(v)

        self._set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

        self._update_viewer_data_on_predict_filter_finished(filt_msk_changed=changed)

        self._set_workflow_state(state=WorkflowState.APPLY_EDITS)

        self._set_message(
            "Prediction finished",
            MessageLevel.CHECK,
        )

    def _on_apply_edits_clicked(self):

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

        for k, v in worker_output.items():

            if k == "merge":

                self.workflow.data["merge"] = v

            else:

                self.workflow.data[k].update(v)

        self._set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

        self._update_viewer_data_on_apply_edits_finished()

        self._set_workflow_state(state=WorkflowState.OVERLAP_ROI)

        self._set_message(
            "Edits applied",
            MessageLevel.CHECK,
        )

    def _on_overlap_clicked(self):

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

    def _start_overlap_thread(self):

        self._start_worker_thread(
            worker_class=OverlapWorker,
            worker_args=(self.workflow.data, self.config, self.workflow.ch_names),
            success_handler=self._on_overlap_finished,
            thread_attr_name="_overlap_worker_thread",
            worker_attr_name="_overlap_worker",
        )

    def _on_overlap_finished(self, worker_output: dict):

        for k, v in worker_output.items():

            if k == "merge":

                self.workflow.data["merge"].update(v)

            elif k == "rois":

                self.workflow.data["rois"] = v

            else:

                self.workflow.data[k].update(v)

        # Delay viewer update
        self._set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

        self._update_viewer_data_on_overlap_finished()

        self._set_viewer_state(ViewerState.LOCKED)

        if self.state.workflow_state == WorkflowState.OVERLAPPING_ROI:

            self._set_workflow_state(state=WorkflowState.OVERLAP_FILTER_OR_SAVE)

        else:

            self._set_workflow_state(
                state=WorkflowState.UPDATE_OVERLAP_FILTER_OR_SAVE, force=True
            )

        self._refresh_selection_summary()

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
            summary_key="summary",
            cnts_key="cnts",
            submsks_area_um2_key="submsks_area_um2",
            pop_order=tuple(self.config["elements"].keys()),
            selected_ids_override=(
                self._manual_selections if self._manual_selections else None
            ),
        )

        self.elem_list = workflow.make_elem_list(elem_records, self.workflow.metadata)

        for k in self.elem_list.keys():

            with open(Path(out_dir_path, f"elem_list_{k}.txt"), "w") as f:
                f.write(self.elem_list[k])

        # Record the old-ID -> new-element-ID / collect-omit correspondence so it
        # is persisted alongside the data.
        self.workflow.data["roi_id_map"] = roi_id_map

        # Write data and metadata to file
        workflow.pickle_data(
            data=self.workflow.data, filename=Path(out_dir_path, "data.pkl")
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

        print(f"Collect / omit summary - {self.workflow.metadata['img_nm']}:\n")

        for p in self._populations_in_config_order():
            ch_summary = self.workflow.data[self.workflow.ch_names[p.primary_ch]][
                "summary"
            ]
            print(
                f"- {self._elem_label(p)}: "
                f"{ch_summary[f'{p.status}_collect']} / {ch_summary[p.status]} "
                f"({ch_summary[f'{p.status}_tube_id']})"
            )

        print("\n")

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
        status_dict = self.workflow.data[ch_nm][f"{ch_other_nm}_status"][status]

        rois = [(roi_id, status_dict.get(roi_id, {})) for roi_id in cnts.keys()]

        if pop_key in self._manual_selections:
            initial_selected = self._manual_selections[pop_key]
        else:
            box = self._get_elem_box(pop_key)
            n_collect = box.spin_n.value()
            initial_selected = set(list(cnts.keys())[:n_collect])

        px_area_um2 = self.workflow.metadata["image"]["px_area_um2"]
        title = self._get_elem_box(pop_key).base_label

        dialog = ChooseROIsWindow(
            title=title,
            rois=rois,
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
        ch_other_nm = self.workflow.ch_names[1 - p.primary_ch]

        cnts = self.workflow.data[ch_nm]["cnts"][status]
        status_dict = self.workflow.data[ch_nm][f"{ch_other_nm}_status"][status]

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
        """Build a message summarising selected ROI count and area per population."""
        lines = ["Selection summary:\n"]
        for p in self._populations_in_config_order():
            n, area = self._pop_selection_summary(p.key)
            base_label = self._get_elem_box(p.key).base_label
            lines.append(f"- {base_label}: {n} ROIs  |  {area:.0f} µm²")

        return "\n".join(lines)

    def _refresh_selection_summary(self):
        """Re-render the per-population selection summary in the message box."""
        self._set_message(self._selection_summary_message(), MessageLevel.CHECK)

    def _on_min_area_changed(self, key: int, value: float):

        self.config["channels"][key]["min_area_um2"] = float(value)

    def _on_min_pct_ovl_changed(self, key: str, value: float):

        self.config[key] = float(value)

    def _on_elem_params_changed(self, key: str, values: dict):

        self.config["elements"][key].update(values)

        self._refresh_selection_summary()

    def _update_elem_boxes(self):

        for p in POPULATIONS:

            box = self.ui.box_elems.get(p.key)

            if box is None:
                continue

            ch_summary = self.workflow.data[self.workflow.ch_names[p.primary_ch]][
                "summary"
            ]
            elem_cfg = self.config["elements"][p.key]

            max_n_collect = ch_summary[p.status]
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
            box.combo_tube.setCurrentText(elem_cfg["tube_id"])
            box.combo_tube.blockSignals(False)

    def _read_elem_boxes(self):

        for p in POPULATIONS:

            box = self.ui.box_elems.get(p.key)
            assert box is not None

            ch_summary = self.workflow.data[self.workflow.ch_names[p.primary_ch]][
                "summary"
            ]

            ch_summary[f"{p.status}_collect"] = box.spin_n.value()
            ch_summary[f"{p.status}_laser_function"] = box.combo_laser.currentText()
            ch_summary[f"{p.status}_tube_id"] = box.combo_tube.currentText()

        self.workflow.data[self.workflow.ch_names[0]]["summary"] = {
            k: self.workflow.data[self.workflow.ch_names[0]]["summary"][k]
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

        self.workflow.data[self.workflow.ch_names[1]]["summary"] = {
            k: self.workflow.data[self.workflow.ch_names[1]]["summary"][k]
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
