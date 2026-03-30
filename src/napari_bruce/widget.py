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
)
from qtpy.QtCore import Signal, QObject, QThread, Qt
from . import configuration
from . import workflow

# Check Java
workflow.require_java()

# Load configuration
config = configuration.get_config()

# Convert config['channels'] keys to int if read from json file
config["channels"] = {int(k): v for k, v in config["channels"].items()}

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

    def __init__(self):

        self.reset()

    def reset(self):

        self.workflow_state = None
        self.viewer_state = None


# %% WorkflowData() ----


class WorkflowData:

    def __init__(self):

        self.reset()

    def reset(self):

        self.path = None
        self.data = None
        self.metadata = None
        self.ch_names = None
        self.channel_mismatch = False


# %% UIComponents() ----


class UIComponents:

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
        self.box_elem_ch0pos_ch1neg = None
        self.box_elem_ch0pos_ch1pos = None
        self.box_elem_ch0pos_ch1amb = None
        self.box_elem_ch1pos_ch0neg = None
        self.box_elem_ch1pos_ch0amb = None


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

        # general layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # label
        self.base_label = label
        self.label = QLabel(label)
        self.label.setStyleSheet("font-weight: bold")
        main_layout.addWidget(self.label)

        # n cells
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("n collect"))

        self.spin_n = QSpinBox()
        self.spin_n.setRange(min_n_collect, max_n_collect)
        self.spin_n.setValue(n_collect)
        row1.addWidget(self.spin_n)

        main_layout.addLayout(row1)

        # tube ID
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("tube ID"))

        self.combo_tube = QComboBox()
        self.combo_tube.addItems(tube_options)
        self.combo_tube.setCurrentText(tube_id)
        row2.addWidget(self.combo_tube)

        main_layout.addLayout(row2)

        # laser function
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("laser function"))

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

        output = {
            "collect": collect,
            "laser_function": self.combo_laser.currentText(),
            "tube_id": self.combo_tube.currentText(),
        }

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

                    stardist_models_dir_path = os.path.join(
                        importlib.resources.files("napari_bruce"), "stardist_models"
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
        out_dir_path = os.path.join(
            Path(self.config["out_dir_path"]).expanduser(), Path(self.path).stem
        )

        ome_tiff_file_path = os.path.join(
            out_dir_path, Path(self.path).stem + ".ome.tiff"
        )

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
        models,
        do_predict=None,
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

        # from csbdeep.utils import normalize

        output = {}
        filt_msk_changed = {}

        for k, v in self.ch_names.items():

            if self.do_predict:

                # Run StarDist
                # img = normalize(
                #     self.data[v]["img"], 1, 99.8, axis=(0, 1)
                # )  # Adjust this depending on training parameters

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
                px_area_um2 = self.metadata["image"]["px_area_um2"]

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

            # Extract submasks and contours
            submsks, cnts = workflow.status_dict_to_submsks_and_cnts(
                msk=self.data[i]["edit_msk"],
                status_dict=k,
                cnt_dict=self.data[i]["edit_cnts"],
            )

            output[i] = {
                f"{j}_status": k,
                "summary": summary,
                "submsks": submsks,
                "cnts": cnts,
            }

        merge_norm_img = self.data["merge"]["merge_norm_img"].copy()

        def rgb_from_config(key):

            return tuple(
                int(x * 255) for x in to_rgba(self.config["elements"][key]["color"])[:3]
            )

        status_colors_ch0 = {
            "neg": rgb_from_config("ch0-pos/ch1-neg"),
            "pos": rgb_from_config("ch0-pos/ch1-pos"),
            "amb": rgb_from_config("ch0-pos/ch1-amb"),
        }

        status_colors_ch1 = {
            "neg": rgb_from_config("ch1-pos/ch0-neg"),
            "amb": rgb_from_config("ch1-pos/ch0-amb"),
        }

        for i, j in status_colors_ch0.items():

            merge_norm_img = cv2.drawContours(
                image=merge_norm_img,
                contours=[x for x in output[ch0_nm]["cnts"][i].values()],
                contourIdx=-1,
                color=j,
                thickness=4,
            )

        for i, j in status_colors_ch1.items():

            merge_norm_img = cv2.drawContours(
                image=merge_norm_img,
                contours=[x for x in output[ch1_nm]["cnts"][i].values()],
                contourIdx=-1,
                color=j,
                thickness=4,
            )

        output["merge"] = {"merge_norm_img_status": merge_norm_img}

        return output


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

        self.build_layout()

        self.init_viewer_layers()

        message = f"""Loading StarDist models...
    
    Channel 0: {self.config['channels'][0]['stardist_model']}
    Channel 1: {self.config['channels'][1]['stardist_model']}
    """

        self.set_message(message, MessageLevel.WORK)

        self.start_worker_thread(
            worker_class=LoadModelWorker,
            worker_args=(self.config,),
            success_handler=self.on_models_loaded,
            thread_attr_name="_load_model_worker_thread",
            worker_attr_name="_load_model_worker",
        )

    def closeEvent(self, event):

        self.cleanup_all_workers()

        super().closeEvent(event)

    def cleanup_all_workers(self):

        thread_attr_names = [
            "_load_model_worker_thread",
            "_load_worker_thread",
            "_predict_filter_worker_thread",
            "_apply_edits_worker_thread",
            "_overlap_worker_thread",
        ]

        for i in thread_attr_names:

            t = getattr(self, i, None)

            if isinstance(t, QThread) and t.isRunning():

                t.quit()
                t.wait()

    def start_worker_thread(
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
            lambda msg, worker=worker_class.__name__: self.on_worker_error(worker, msg)
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

    def on_worker_error(self, worker_name: str, error_msg: str):

        self.workflow.channel_mismatch = False

        self.set_message(
            f"{worker_name} failed with the following error:\n\n{error_msg}",
            MessageLevel.ERROR,
        )

        self.ui.btn_clear.setEnabled(True)

        return

    def build_layout(self):

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
        self.header_layout.addWidget(self.msg_container)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.header_layout)
        self.layout.addStretch(1)

        self.setLayout(self.layout)
        self.setMinimumWidth(300)
        self.setMaximumWidth(1000)

    def init_viewer_layers(self):

        self.layers = {}

        image_array = np.zeros((1, 1), dtype=np.uint8)
        label_array = np.zeros((1, 1), dtype=np.uint16)
        merge_array = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.viewer.layers.events.blocker():

            for i in [0, 1]:

                self.layers[i] = {}

                self.layers[i]["image"] = self.viewer.add_image(
                    image_array, name=f"ch{i} normalized image", visible=False
                )

            for i in [0, 1]:

                color_dict = {
                    0: (0.0, 0.0, 0.0, 0.0),
                    None: to_rgba(self.config["channels"][i]["color"]),
                }

                self.layers[i]["labels"] = self.viewer.add_labels(
                    label_array,
                    name=f"ch{i} - remove",
                    opacity=0.4,
                    colormap=color_dict,
                    visible=True,
                    rendering="iso_categorical",
                )

                self.layers[i]["labels"].visible = False
                self.layers[i]["labels"].contrast_limits = (0, 65535)
                self.layers[i]["labels"].brush_size = 60

            for i in [0, 1]:

                self.layers[i]["shapes"] = self.viewer.add_shapes(
                    data=[],
                    name=f"ch{i} - add",
                    shape_type="path",
                    edge_color=self.config["channels"][i]["color"],
                    edge_width=6,
                    visible=False,
                )

            self.layers["merge"] = self.viewer.add_image(
                merge_array, name="Merge + ROIs", rgb=True, visible=False
            )

    def reset_viewer_layers(self):

        image_array = np.zeros((1, 1), dtype=np.uint8)
        label_array = np.zeros((1, 1), dtype=np.uint16)
        merge_array = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.viewer.layers.events.blocker():

            for i in [0, 1]:

                self.layers[i]["image"].data = image_array
                self.layers[i]["image"].name = f"ch{i} normalized image"

                self.layers[i]["labels"].visible = False
                self.layers[i]["labels"].data = label_array
                self.layers[i]["labels"].name = f"ch{i} - remove"
                self.layers[i]["labels"].contour = 0
                self.layers[i]["labels"].opacity = 0.4
                self.layers[i]["labels"].contrast_limits = (0, 65535)
                self.layers[i]["labels"].brush_size = 60

                self.layers[i]["shapes"].visible = False
                self.layers[i]["shapes"].data = []
                self.layers[i]["shapes"].name = f"ch{i} - add"
                self.layers[i]["shapes"].edge_color = self.config["channels"][i][
                    "color"
                ]
                self.layers[i]["shapes"].edge_width = 6

            self.layers["merge"].data = merge_array
            self.layers["merge"].name = "Merge + ROIs"

    def set_workflow_state(self, state: WorkflowState, force=False):

        if not force and state == self.state.workflow_state:

            return

        self.state.workflow_state = state

        self.update_viewer_from_workflow()
        self.update_ui_from_workflow()

    def update_viewer_from_workflow(self):

        workflow_state = self.state.workflow_state

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

        self.set_viewer_state(state=viewer_state)

    def set_viewer_state(self, state: ViewerState):

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

                self.layers["merge"].visible = True

                self.viewer.layers.selection.active = self.layers["merge"]

    def update_ui_from_workflow(self):

        workflow_state = self.state.workflow_state

        if workflow_state == WorkflowState.SELECT_FILE:

            self.set_select_file_ui()

        elif workflow_state == WorkflowState.LOAD_IMAGE:

            self.set_load_image_ui()

        elif workflow_state in {
            WorkflowState.LOADING_IMAGE,
            WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI,
        }:

            self.set_loading_image_ui()

        elif workflow_state == WorkflowState.PREDICT_ROI:

            self.set_predict_roi_ui(loaded_for_predict=False)

        elif workflow_state == WorkflowState.IMAGE_LOADED_FOR_PREDICT_ROI:

            self.set_predict_roi_ui(loaded_for_predict=True)

        elif workflow_state == WorkflowState.PREDICTING_ROI:

            self.set_predicting_roi_ui()

        elif workflow_state == WorkflowState.APPLY_EDITS:

            self.set_apply_edits_ui()

        elif workflow_state == WorkflowState.APPLYING_EDITS:

            self.set_applying_edits_ui()

        elif workflow_state == WorkflowState.OVERLAP_ROI:

            self.set_overlap_roi_ui()

        elif workflow_state == WorkflowState.OVERLAPPING_ROI:

            self.set_overlapping_roi_ui()

        elif workflow_state == WorkflowState.OVERLAP_FILTER_OR_SAVE:

            self.set_overlap_filter_or_save_ui(build_ui=True)

        elif workflow_state == WorkflowState.UPDATE_OVERLAP_FILTER_OR_SAVE:

            self.set_overlap_filter_or_save_ui(build_ui=False)

        elif workflow_state == WorkflowState.CLEARED:

            self.set_cleared_ui()

        if workflow_state in {
            WorkflowState.LOADING_IMAGE,
            WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI,
            WorkflowState.PREDICTING_ROI,
            WorkflowState.APPLYING_EDITS,
            WorkflowState.OVERLAPPING_ROI,
        }:

            self.set_ui_enabled(False)

    def set_ui_enabled(self, enabled: bool):

        for i in vars(self.ui).values():

            if isinstance(i, QWidget):

                i.setEnabled(enabled)

    def set_select_file_ui(self):

        self.ui.btn_select = QPushButton("Select file")
        self.ui.btn_select.clicked.connect(self.on_select_clicked)
        self.header_layout.insertWidget(0, self.ui.btn_select)

    def set_load_image_ui(self):

        # Remove 'Select file' button from viewer
        self.layout.removeWidget(self.ui.btn_select)
        self.ui.btn_select.hide()
        self.ui.btn_select.deleteLater()
        self.ui.btn_select = None

        # Add 'Load images', 'Predict cells' and 'Clear' buttons to viewer
        self.ui.btn_load = QPushButton("Load images")
        self.ui.btn_load.clicked.connect(self.on_load_clicked)

        self.ui.btn_predict = QPushButton("Predict cells")
        self.ui.btn_predict.clicked.connect(self.on_predict_clicked)

        self.ui.btn_clear = QPushButton("Clear")
        self.ui.btn_clear.clicked.connect(lambda checked=False: self.on_clear_clicked())

        for i, j in zip(
            [0, 1, 2], [self.ui.btn_load, self.ui.btn_predict, self.ui.btn_clear]
        ):

            self.layout.insertWidget(i, j)

    def set_loading_image_ui(self):

        # Remove 'Load images' button from viewer
        self.layout.removeWidget(self.ui.btn_load)
        self.ui.btn_load.hide()
        self.ui.btn_load.deleteLater()
        self.ui.btn_load = None

    def set_predict_roi_ui(self, loaded_for_predict=False):

        # Add 'min n pix' boxes to viewer
        self.ui.box_min_area_ch0 = ParamValueBox(
            label=f"min area (\u00b5m\u00b2) {self.workflow.ch_names[0]}",
            default=self.config["channels"][0]["min_area_um2"],
            min_val=0.0,
            max_val=10000.0,
        )
        self.ui.box_min_area_ch0.valueChanged.connect(
            lambda x: self.on_min_area_changed(0, x)
        )

        self.ui.box_min_area_ch1 = ParamValueBox(
            label=f"min area (\u00b5m\u00b2) {self.workflow.ch_names[1]}",
            default=self.config["channels"][1]["min_area_um2"],
            min_val=0.0,
            max_val=10000.0,
        )
        self.ui.box_min_area_ch1.valueChanged.connect(
            lambda x: self.on_min_area_changed(1, x)
        )

        for i, j in zip([0, 1], [self.ui.box_min_area_ch0, self.ui.box_min_area_ch1]):

            self.layout.insertWidget(i, j)

        if loaded_for_predict:

            # Disable 'min n pix' boxes
            for i in [self.ui.box_min_area_ch0, self.ui.box_min_area_ch1]:

                i.setEnabled(False)

        else:

            # Re-enable 'Predict cells' and 'Clear' buttons
            for i in [self.ui.btn_predict, self.ui.btn_clear]:

                i.setEnabled(True)

    def set_predicting_roi_ui(self):

        for i in ["btn_load", "btn_predict"]:

            j = getattr(self.ui, i, None)

            if j is not None:

                self.layout.removeWidget(j)
                j.hide()
                j.deleteLater()
                setattr(self.ui, i, None)

    def set_apply_edits_ui(self):

        # Add 'Adjust size filter' and 'Apply edits' buttons to viewer if predictions are returned for the first time
        self.ui.btn_filter_size = QPushButton("Adjust size filter")
        self.ui.btn_filter_size.clicked.connect(self.on_filter_size_clicked)

        self.ui.btn_apply_edits = QPushButton("Apply edits")
        self.ui.btn_apply_edits.clicked.connect(self.on_apply_edits_clicked)

        for i, j in zip([2, 3], [self.ui.btn_filter_size, self.ui.btn_apply_edits]):

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

            i.setEnabled(True)

    def set_applying_edits_ui(self):

        # Remove 'min n pix' boxes, 'Adjust size filter' and 'Apply edits' buttons from viewer
        for i in [
            "box_min_area_ch0",
            "box_min_area_ch1",
            "btn_filter_size",
            "btn_apply_edits",
        ]:

            j = getattr(self.ui, i, None)
            self.layout.removeWidget(j)
            j.hide()
            j.deleteLater()
            setattr(self.ui, i, None)

    def set_overlap_roi_ui(self):

        # Add 'min % ovl' boxes and 'Find overlaps' button to viewer
        self.ui.box_min_pct_ovl_ch0_by_ch1 = ParamValueBox(
            label=f"min % overlap {self.workflow.ch_names[0]} / {self.workflow.ch_names[1]}",
            default=self.config["min_pct_ovl_ch0_by_ch1"],
            min_val=0.0,
            max_val=100.0,
        )
        self.ui.box_min_pct_ovl_ch0_by_ch1.valueChanged.connect(
            lambda x: self.on_min_pct_ovl_changed("min_pct_ovl_ch0_by_ch1", x)
        )

        self.ui.box_min_pct_ovl_ch1_by_ch0 = ParamValueBox(
            label=f"min % overlap {self.workflow.ch_names[1]} / {self.workflow.ch_names[0]}",
            default=self.config["min_pct_ovl_ch1_by_ch0"],
            min_val=0.0,
            max_val=100.0,
        )
        self.ui.box_min_pct_ovl_ch1_by_ch0.valueChanged.connect(
            lambda x: self.on_min_pct_ovl_changed("min_pct_ovl_ch1_by_ch0", x)
        )

        self.ui.btn_overlap = QPushButton("Find overlaps")
        self.ui.btn_overlap.clicked.connect(self.on_overlap_clicked)

        for i, j in zip(
            [0, 1, 2],
            [
                self.ui.box_min_pct_ovl_ch0_by_ch1,
                self.ui.box_min_pct_ovl_ch1_by_ch0,
                self.ui.btn_overlap,
            ],
        ):

            self.layout.insertWidget(i, j)

        # Re-enable 'Clear' button
        self.ui.btn_clear.setEnabled(True)

    def set_overlapping_roi_ui(self):

        # Remove 'Find overlaps' button from viewer
        self.layout.removeWidget(self.ui.btn_overlap)
        self.ui.btn_overlap.hide()
        self.ui.btn_overlap.deleteLater()
        self.ui.btn_overlap = None

    def set_overlap_filter_or_save_ui(self, build_ui=True):

        if build_ui:

            # Add 'Adjust overlap filter' button, 'element' boxes and 'Save results' button to viewer if overlaps are returned for the first time
            self.ui.btn_overlap_filter = QPushButton("Adjust overlap filter")
            self.ui.btn_overlap_filter.clicked.connect(self.on_overlap_filter_clicked)

            self.ui.btn_save = QPushButton("Save results")
            self.ui.btn_save.clicked.connect(self.on_save_clicked)

            plus = "\u207a"
            minus = "\u207b"

            self.ui.box_elem_ch0pos_ch1neg = ElementConfigBox(
                label=f"{self.workflow.ch_names[0]}{plus} / {self.workflow.ch_names[1]}{minus}"
            )

            self.ui.box_elem_ch0pos_ch1neg.valueChanged.connect(
                lambda x: self.on_elem_params_changed("ch0-pos/ch1-neg", x)
            )

            self.ui.box_elem_ch0pos_ch1pos = ElementConfigBox(
                label=f"{self.workflow.ch_names[0]}{plus} / {self.workflow.ch_names[1]}{plus}"
            )

            self.ui.box_elem_ch0pos_ch1pos.valueChanged.connect(
                lambda x: self.on_elem_params_changed("ch0-pos/ch1-pos", x)
            )

            self.ui.box_elem_ch0pos_ch1amb = ElementConfigBox(
                label=f"{self.workflow.ch_names[0]}{plus} / {self.workflow.ch_names[1]}-amb"
            )

            self.ui.box_elem_ch0pos_ch1amb.valueChanged.connect(
                lambda x: self.on_elem_params_changed("ch0-pos/ch1-amb", x)
            )

            self.ui.box_elem_ch1pos_ch0neg = ElementConfigBox(
                label=f"{self.workflow.ch_names[1]}{plus} / {self.workflow.ch_names[0]}{minus}"
            )

            self.ui.box_elem_ch1pos_ch0neg.valueChanged.connect(
                lambda x: self.on_elem_params_changed("ch1-pos/ch0-neg", x)
            )

            self.ui.box_elem_ch1pos_ch0amb = ElementConfigBox(
                label=f"{self.workflow.ch_names[1]}{plus} / {self.workflow.ch_names[0]}-amb"
            )

            self.ui.box_elem_ch1pos_ch0amb.valueChanged.connect(
                lambda x: self.on_elem_params_changed("ch1-pos/ch0-amb", x)
            )

            for i, j in zip(
                [2, 3, 4, 5, 6, 7, 8],
                [
                    self.ui.btn_overlap_filter,
                    self.ui.box_elem_ch0pos_ch1neg,
                    self.ui.box_elem_ch0pos_ch1pos,
                    self.ui.box_elem_ch0pos_ch1amb,
                    self.ui.box_elem_ch1pos_ch0neg,
                    self.ui.box_elem_ch1pos_ch0amb,
                    self.ui.btn_save,
                ],
            ):

                self.layout.insertWidget(i, j)

        # Re-enable 'min % ovl' and 'element' boxes, 'Adjust overlap filter', 'Save results' and 'Clear' buttons
        # => 'Adjust overlap filter' button, 'element' boxes and 'Save results' button do not exist yet if overlaps are returned for the first time
        for i in [
            self.ui.box_min_pct_ovl_ch0_by_ch1,
            self.ui.box_min_pct_ovl_ch1_by_ch0,
            self.ui.btn_overlap_filter,
            self.ui.box_elem_ch0pos_ch1neg,
            self.ui.box_elem_ch0pos_ch1pos,
            self.ui.box_elem_ch0pos_ch1amb,
            self.ui.box_elem_ch1pos_ch0neg,
            self.ui.box_elem_ch1pos_ch0amb,
            self.ui.btn_save,
            self.ui.btn_clear,
        ]:

            i.setEnabled(True)

        # Update 'element' boxes
        self.update_elem_boxes()

    def set_cleared_ui(self):

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
            "box_elem_ch0pos_ch1neg",
            "box_elem_ch0pos_ch1pos",
            "box_elem_ch0pos_ch1amb",
            "box_elem_ch1pos_ch0neg",
            "box_elem_ch1pos_ch0amb",
            "btn_save",
        ]:

            j = getattr(self.ui, i, None)

            if j is not None:

                self.layout.removeWidget(j)
                j.hide()
                j.deleteLater()
                setattr(self.ui, i, None)

    def update_viewer_data_on_load_finished(self):

        with self.viewer.layers.events.blocker():

            # Add normalized images to viewer
            for k, v in self.workflow.ch_names.items():

                self.layers[k]["image"].data = self.workflow.data[v]["norm_img"]
                self.layers[k]["image"].name = f"{v} normalized image"

                self.layers[k]["labels"].name = f"{v} - remove"

                self.layers[k]["shapes"].name = f"{v} - add"

            self.layers["merge"].data = np.zeros(
                self.workflow.data[self.workflow.ch_names[0]]["norm_img"].shape + (3,),
                dtype=np.uint8,
            )

        self.viewer.reset_view()

    def update_viewer_data_on_predict_filter_finished(self, filt_msk_changed: dict):

        for k, v in self.workflow.ch_names.items():

            if filt_msk_changed[v]:

                labels = self.layers[k]["labels"]

                with labels.events.blocker():

                    labels.data = self.workflow.data[v]["filt_msk"]

    def update_viewer_data_on_apply_edits_finished(self):

        with self.viewer.layers.events.blocker():

            self.layers["merge"].data = self.workflow.data["merge"][
                "merge_norm_img_rois"
            ]

    def update_viewer_data_on_overlap_finished(self):

        with self.viewer.layers.events.blocker():

            self.layers["merge"].data = self.workflow.data["merge"][
                "merge_norm_img_status"
            ]

    def set_message(self, text: str, level: MessageLevel = MessageLevel.NONE):

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

    def on_models_loaded(self, models: dict):

        self.models = models

        self.set_message("", MessageLevel.NONE)

        self.set_workflow_state(WorkflowState.SELECT_FILE)

    def on_select_clicked(self):

        # Record user-selected file path
        self.workflow.path = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select .zvi file",
            directory=str(Path(self.config["in_dir_path"]).expanduser()),
            filter="Images (*.zvi)",
        )[0]

        # Abort if user cancelled selection
        if self.workflow.path == "":

            self.set_message(".zvi file selection cancelled", MessageLevel.WARNING)

            return

        self.set_workflow_state(state=WorkflowState.LOAD_IMAGE)

        self.set_message(".zvi file selected", MessageLevel.INFO)

    def on_clear_clicked(self, *args):

        # Reset attributes to default values
        self.config = copy.deepcopy(config)

        self.set_workflow_state(state=WorkflowState.CLEARED)

        self.workflow.reset()

        self.ui.reset()

        self.state.reset()

        self.reset_viewer_layers()

        self.set_workflow_state(state=WorkflowState.SELECT_FILE)

        self.set_message("", MessageLevel.NONE)

    def on_load_clicked(self):

        self.set_workflow_state(state=WorkflowState.LOADING_IMAGE)

        self.set_message("Loading images...", MessageLevel.WORK)

        # Trigger load worker thread
        self.start_load_thread()

    def on_predict_clicked(self):

        # Trigger load worker thread if 'Predict cells' button was clicked directly
        if self.state.workflow_state == WorkflowState.LOAD_IMAGE:

            # Record that prediction has been requested
            self.set_workflow_state(state=WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI)

            self.set_message("Loading images...", MessageLevel.WORK)

            # Trigger load worker thread
            self.start_load_thread()

        elif self.state.workflow_state == WorkflowState.PREDICT_ROI:

            # Alternatively, proceed with prediction if 'Load images' button was clicked previously
            self.set_workflow_state(state=WorkflowState.PREDICTING_ROI)

            self.on_img_loaded_and_predict_clicked()

    def start_load_thread(self):

        self.start_worker_thread(
            worker_class=LoadWorker,
            worker_args=(self.workflow.path, self.config),
            success_handler=self.on_load_finished,
            thread_attr_name="_load_worker_thread",
            worker_attr_name="_load_worker",
        )

    def on_load_finished(self, worker_output: dict):

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

        self.set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

        self.update_viewer_data_on_load_finished()

        # Stop here if 'Load images' button was clicked
        if self.state.workflow_state == WorkflowState.LOADING_IMAGE:

            self.set_workflow_state(state=WorkflowState.PREDICT_ROI)

            self.set_message(
                "Check images in viewer\n\nAdjust min size thresholds (optional)\n\nClic 'Predict cells' when ready",
                MessageLevel.CHECK,
            )

        # Alternatively, proceed with prediction if 'Predict cells' button was clicked
        elif self.state.workflow_state == WorkflowState.LOADING_IMAGE_FOR_PREDICT_ROI:

            self.set_workflow_state(state=WorkflowState.IMAGE_LOADED_FOR_PREDICT_ROI)

            # Proceed with prediction
            self.on_img_loaded_and_predict_clicked()

    def on_img_loaded_and_predict_clicked(self):

        self.set_workflow_state(state=WorkflowState.PREDICTING_ROI)

        self.set_message("Predicting ROIs...", MessageLevel.WORK)

        # Trigger predict worker thread
        self.start_predict_filter_thread(do_predict=True)

    def start_predict_filter_thread(self, do_predict=True):

        self.start_worker_thread(
            worker_class=PredictFilterWorker,
            worker_args=(
                self.workflow.data,
                self.workflow.metadata,
                self.config,
                self.workflow.ch_names,
                self.models,
                do_predict,
            ),
            success_handler=self.on_predict_filter_finished,
            thread_attr_name="_predict_filter_worker_thread",
            worker_attr_name="_predict_filter_worker",
        )

    def on_filter_size_clicked(self):

        self.start_predict_filter_thread(do_predict=False)

    def on_predict_filter_finished(self, worker_output: dict):

        output = worker_output["output"]
        changed = worker_output["changed"]

        for k, v in output.items():

            self.workflow.data[k].update(v)

        self.set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

        self.update_viewer_data_on_predict_filter_finished(filt_msk_changed=changed)

        self.set_workflow_state(state=WorkflowState.APPLY_EDITS)

        self.set_message(
            "Check images in viewer\n\nEdit cell contours (optional)\n\nClic 'Apply edits' when finished",
            MessageLevel.CHECK,
        )

    def on_apply_edits_clicked(self):

        self.set_workflow_state(state=WorkflowState.APPLYING_EDITS)

        self.set_message("Applying edits...", MessageLevel.WORK)

        # Trigger apply edits worker thread
        self.start_apply_edits_thread()

    def start_apply_edits_thread(self):

        copied_masks = {}
        copied_shapes = {}

        for k, v in self.workflow.ch_names.items():

            copied_masks[v] = self.layers[k]["labels"].data.copy()
            copied_shapes[v] = self.layers[k]["shapes"].data

        self.start_worker_thread(
            worker_class=ApplyEditsWorker,
            worker_args=(
                copied_masks,
                copied_shapes,
                self.workflow.data,
                self.workflow.metadata,
                self.config,
                self.workflow.ch_names,
            ),
            success_handler=self.on_apply_edits_finished,
            thread_attr_name="_apply_edits_worker_thread",
            worker_attr_name="_apply_edits_worker",
        )

    def on_apply_edits_finished(self, worker_output: dict):

        for k, v in worker_output.items():

            if k == "merge":

                self.workflow.data["merge"] = v

            else:

                self.workflow.data[k].update(v)

        self.set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

        self.update_viewer_data_on_apply_edits_finished()

        self.set_workflow_state(state=WorkflowState.OVERLAP_ROI)

        self.set_message(
            "Adjust overlap thresholds (optional)\n\nClic 'Find overlaps' when ready",
            MessageLevel.CHECK,
        )

    def on_overlap_clicked(self):

        self.set_workflow_state(state=WorkflowState.OVERLAPPING_ROI)

        self.set_message("Computing overlaps...", MessageLevel.WORK)

        # Trigger overlap worker thread
        self.start_overlap_thread()

    def on_overlap_filter_clicked(self):

        self.set_message("Re-computing overlaps...", MessageLevel.WORK)

        # Trigger overlap worker thread
        self.start_overlap_thread()

    def start_overlap_thread(self):

        self.start_worker_thread(
            worker_class=OverlapWorker,
            worker_args=(self.workflow.data, self.config, self.workflow.ch_names),
            success_handler=self.on_overlap_finished,
            thread_attr_name="_overlap_worker_thread",
            worker_attr_name="_overlap_worker",
        )

    def on_overlap_finished(self, worker_output: dict):

        for k, v in worker_output.items():

            if k == "merge":

                self.workflow.data["merge"].update(v)

            else:

                self.workflow.data[k].update(v)

        # Delay viewer update
        self.set_message("Updating viewer...", MessageLevel.BUSY)

        QApplication.processEvents()

        self.update_viewer_data_on_overlap_finished()

        self.set_viewer_state(ViewerState.LOCKED)

        if self.state.workflow_state == WorkflowState.OVERLAPPING_ROI:

            self.set_workflow_state(state=WorkflowState.OVERLAP_FILTER_OR_SAVE)

        else:

            self.set_workflow_state(
                state=WorkflowState.UPDATE_OVERLAP_FILTER_OR_SAVE, force=True
            )

        self.set_message(
            "Adjust element list parameters (optional)\n\nClic 'Save results' when finished",
            MessageLevel.CHECK,
        )

    def on_save_clicked(self):

        # Read content of 'element' boxes
        self.read_elem_boxes()

        # Define output directory path
        out_dir_path = os.path.join(
            Path(self.config["out_dir_path"]).expanduser(),
            self.workflow.metadata["img_nm"],
        )

        # Construct element list and write to file
        self.elem_list = workflow.make_elem_txt(
            data_dict=self.workflow.data,
            metadata_dict=self.workflow.metadata,
            config_dict=self.config,
            summary_key="summary",
            cnts_key="cnts",
            submsks_area_um2_key="submsks_area_um2",
        )

        with open(os.path.join(out_dir_path, "elem_list.txt"), "w") as f:
            f.write(self.elem_list)

        # Write data and metadata to file
        workflow.pickle_data(
            data=self.workflow.data, filename=os.path.join(out_dir_path, "data.pkl")
        )

        workflow.pickle_data(
            data=self.workflow.metadata,
            filename=os.path.join(out_dir_path, "metadata.pkl"),
        )

        # Write config to file
        with open(os.path.join(out_dir_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

        self.set_message(
            f"Results saved at:\n{self.config['out_dir_path']}\n\nIn subfolder:\n{self.workflow.metadata['img_nm']}",
            MessageLevel.SAVE,
        )

    def on_min_area_changed(self, key: int, value: float):

        self.config["channels"][key]["min_area_um2"] = float(value)

    def on_min_pct_ovl_changed(self, key: str, value: float):

        self.config[key] = float(value)

    def on_elem_params_changed(self, key: str, values: dict):

        self.config["elements"][key].update(values)

    def update_elem_boxes(self):

        primary_channel_id = [0, 0, 0, 1, 1]

        secondary_channel_status = ["neg", "pos", "amb", "neg", "amb"]

        population = [
            "ch0-pos/ch1-neg",
            "ch0-pos/ch1-pos",
            "ch0-pos/ch1-amb",
            "ch1-pos/ch0-neg",
            "ch1-pos/ch0-amb",
        ]

        box = [
            self.ui.box_elem_ch0pos_ch1neg,
            self.ui.box_elem_ch0pos_ch1pos,
            self.ui.box_elem_ch0pos_ch1amb,
            self.ui.box_elem_ch1pos_ch0neg,
            self.ui.box_elem_ch1pos_ch0amb,
        ]

        for i, j, k, l in zip(
            primary_channel_id, secondary_channel_status, population, box
        ):

            max_n_collect = self.workflow.data[self.workflow.ch_names[i]]["summary"][j]
            n_collect = (
                max_n_collect if self.config["elements"][k]["collect"] is True else 0
            )
            laser_function = (
                self.config["elements"][k]["laser_function"]
                if max_n_collect > 0
                else ""
            )
            tube_id = self.config["elements"][k]["tube_id"] if max_n_collect > 0 else ""
            color = self.config["elements"][k]["color"]

            if l is not None:

                base_label = l.base_label
                label = l.label
                label.setText(f"{base_label} ({max_n_collect})")
                label.setStyleSheet(f"font-weight: bold; color: {color}")

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

        secondary_channel_status = ["neg", "pos", "amb", "neg", "amb"]

        box = [
            self.ui.box_elem_ch0pos_ch1neg,
            self.ui.box_elem_ch0pos_ch1pos,
            self.ui.box_elem_ch0pos_ch1amb,
            self.ui.box_elem_ch1pos_ch0neg,
            self.ui.box_elem_ch1pos_ch0amb,
        ]

        for i, j, k in zip(primary_channel_id, secondary_channel_status, box):

            n_collect = k.spin_n.value()
            laser_function = k.combo_laser.currentText()
            tube_id = k.combo_tube.currentText()

            self.workflow.data[self.workflow.ch_names[i]]["summary"][
                f"{j}_collect"
            ] = n_collect

            self.workflow.data[self.workflow.ch_names[i]]["summary"][
                f"{j}_laser_function"
            ] = laser_function

            self.workflow.data[self.workflow.ch_names[i]]["summary"][
                f"{j}_tube_id"
            ] = tube_id

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
