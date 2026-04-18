# %% Set up ----

import pickle
import traceback
from pathlib import Path

import numpy as np
from enum import Enum
from qtpy.QtCore import QEvent, QObject, QThread, Qt, Signal
from qtpy.QtGui import QFontMetrics
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from . import workflow
from . import configuration

# Get channel-specific ROI colors from configuration
config = configuration.get_config()
CHANNEL_COLORS = [
    config["channels_annotation"][0]["color"],
    config["channels_annotation"][1]["color"],
]

# %% MessageLevel ----


class MessageLevel(Enum):

    NONE = None
    INFO = QStyle.SP_MessageBoxInformation
    BUSY = QStyle.SP_BrowserReload
    WORK = QStyle.SP_ComputerIcon
    CHECK = QStyle.SP_ArrowRight
    SAVE = QStyle.SP_DialogSaveButton
    WARNING = QStyle.SP_MessageBoxWarning
    ERROR = QStyle.SP_MessageBoxCritical


# %% LoadPklWorker ----


class LoadPklWorker(QObject):

    sig_success = Signal(object)
    sig_error = Signal(str)

    def __init__(self, in_dir_path: Path, out_dir_path: Path, parent=None):

        super().__init__(parent)
        self.in_dir_path = in_dir_path
        self.out_dir_path = out_dir_path

    def run(self):

        try:

            annotated_pkl = self.out_dir_path / "imgs_annotated.pkl"

            if annotated_pkl.exists():

                with open(annotated_pkl, "rb") as f:
                    imgs = pickle.load(f)

            else:

                with open(self.in_dir_path / "imgs.pkl", "rb") as f:
                    imgs = pickle.load(f)

            self.sig_success.emit(imgs)

        except Exception as e:

            self.sig_error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# %% _CloseEventFilter ----


class _CloseEventFilter(QObject):

    def __init__(self, owner: "AnnotationManager", parent=None):

        super().__init__(parent)
        self._owner = owner

    def eventFilter(self, obj, event):

        if event.type() == QEvent.Close:

            if self._owner._unsaved_changes:

                reply = QMessageBox.question(
                    self._owner.viewer.window._qt_window,
                    "Unsaved annotations",
                    "You have unsaved annotations.\nSave before closing?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                    QMessageBox.Save,
                )

                if reply == QMessageBox.Save:

                    self._owner._on_save_clicked()

                elif reply == QMessageBox.Cancel:

                    event.ignore()
                    return True

        return super().eventFilter(obj, event)


# %% AnnotationManager ----


class AnnotationManager(QWidget):

    def __init__(
        self,
        viewer: "napari.Viewer",
        in_dir_path: str,
        out_dir_path: str,
    ):

        super().__init__()

        self.viewer = viewer
        self.in_dir_path = Path(in_dir_path).expanduser()
        self.out_dir_path = Path(out_dir_path).expanduser()
        self.out_dir_path.mkdir(parents=True, exist_ok=True)

        self.imgs = {}
        self.img_keys = []
        self.n_imgs = 0
        self.idx = 0

        self.img_layers = {}
        self.shape_layers = {}

        self._unsaved_changes = False
        self._loading = False

        self._close_filter = _CloseEventFilter(owner=self)
        self.viewer.window._qt_window.installEventFilter(self._close_filter)

        self._build_layout()
        self._start_load_thread()

    def closeEvent(self, event):

        t = getattr(self, "_load_thread", None)

        if isinstance(t, QThread) and t.isRunning():

            t.quit()
            t.wait()

        super().closeEvent(event)

    # %% Layout ----

    def _build_layout(self):

        self.lbl_counter = QLabel("")
        self.lbl_counter.setAlignment(Qt.AlignCenter)
        self.lbl_counter.setWordWrap(True)
        self.lbl_counter.setStyleSheet("font-weight: bold")

        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self._on_prev_clicked)
        self.btn_prev.setEnabled(False)

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._on_next_clicked)
        self.btn_next.setEnabled(False)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_save.setEnabled(False)

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

        layout = QVBoxLayout()
        layout.addWidget(self.lbl_counter)
        layout.addLayout(nav_row)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.msg_container)

        self.setLayout(layout)
        self.setMinimumWidth(300)
        self.setMaximumWidth(1000)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _update_counter(self):

        if self.n_imgs == 0:

            self.lbl_counter.setText("")

            return

        fn = self.img_keys[self.idx]
        self.lbl_counter.setText(f"Image {self.idx + 1} / {self.n_imgs}\n{fn}")

    def _update_nav_buttons(self):

        self.btn_prev.setEnabled(self.idx > 0)
        self.btn_next.setEnabled(self.idx < self.n_imgs - 1)

    # %% Loading ----

    def _start_load_thread(self):

        self._load_thread = QThread()

        self._load_worker = LoadPklWorker(
            in_dir_path=self.in_dir_path,
            out_dir_path=self.out_dir_path,
        )

        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.sig_success.connect(self._on_load_finished)
        self._load_worker.sig_error.connect(self._on_load_error)
        self._load_worker.sig_success.connect(self._load_thread.quit)
        self._load_worker.sig_error.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    def _on_load_finished(self, imgs: dict):

        self.imgs = imgs
        self.img_keys = list(imgs.keys())
        self.n_imgs = len(self.img_keys)

        if self.n_imgs == 0:

            print("No images found in imgs.pkl.")

            return

        self._init_viewer_layers()
        self._load_image(0)

        self.btn_save.setEnabled(True)

    def _on_load_error(self, error_msg: str):

        print(f"Failed to load imgs.pkl:\n\n{error_msg}")

    # %% Viewer layers ----

    def _init_viewer_layers(self):

        ch_names = list(self.imgs[self.img_keys[0]]["data"].keys())
        image_array = np.zeros((1, 1), dtype=np.uint8)

        with self.viewer.layers.events.blocker():

            # Image layers added first — appear below shapes in the layer list
            for cn in ch_names:

                self.img_layers[cn] = self.viewer.add_image(
                    image_array,
                    name=f"{cn} - image",
                    visible=True,
                )

            # Shapes layers added second — appear above images in the layer list
            for i, (cn, col) in enumerate(zip(ch_names, CHANNEL_COLORS)):

                layer = self.viewer.add_shapes(
                    data=[],
                    name=f"{cn} - ROIs",
                    edge_color=col,
                    face_color="transparent",
                    edge_width=6,
                    visible=True,
                )
                layer.current_face_color = "transparent"
                layer.current_edge_color = col
                layer.current_edge_width = 6
                layer.events.data.connect(self._on_shapes_changed)
                self.shape_layers[cn] = layer

    def _on_shapes_changed(self, *args, **kwargs):

        if not self._loading:

            self._unsaved_changes = True

    def _load_image(self, idx: int):

        fn = self.img_keys[idx]
        ch_names_data = list(self.imgs[fn]["data"].keys())

        self._loading = True

        with self.viewer.layers.events.blocker():

            for i, (cn_layer, cn_data) in enumerate(
                zip(self.img_layers, ch_names_data)
            ):

                data = self.imgs[fn]["data"][cn_data]

                self.img_layers[cn_layer].data = data["norm_img"]
                self.img_layers[cn_layer].name = f"{cn_data} - image"
                self.img_layers[cn_layer].reset_contrast_limits()

                existing_rois = data.get("rois", [])

                layer = self.shape_layers[cn_layer]
                layer.name = f"{cn_data} - ROIs"
                layer.data = (
                    [arr.copy() for arr in existing_rois] if existing_rois else []
                )

                if existing_rois:
                    layer.selected_data = set(range(len(existing_rois)))
                    layer.edge_width = 6
                    layer.selected_data = set()

                layer.current_face_color = "transparent"
                layer.current_edge_color = CHANNEL_COLORS[i]
                layer.current_edge_width = 6
                layer.mode = "add_path"

        self._loading = False

        self.viewer.reset_view()

        self.idx = idx
        self._update_counter()
        self._update_nav_buttons()

    # %% Annotation save / convert ----

    def _save_current_annotations(self):
        """Save shapes layer data back into imgs dict and convert to label masks."""

        fn = self.img_keys[self.idx]
        ch_names_data = list(self.imgs[fn]["data"].keys())

        for cn_layer, cn_data in zip(self.shape_layers, ch_names_data):

            layer = self.shape_layers[cn_layer]
            rois = [arr.copy() for arr in layer.data]

            self.imgs[fn]["data"][cn_data]["rois"] = rois

            img_shape = self.imgs[fn]["data"][cn_data]["img"].shape
            empty_msk = np.zeros(img_shape, dtype=np.uint16)

            if rois:

                msk = workflow.append_shapes_to_msk(
                    msk=empty_msk,
                    shapes=rois,
                    start_idx=1,
                )
                self.imgs[fn]["data"][cn_data]["msk"] = msk.astype(np.uint16)

            else:

                self.imgs[fn]["data"][cn_data]["msk"] = empty_msk

    # %% Message ----

    def set_message(self, text: str, level: MessageLevel = MessageLevel.NONE):

        self.msg_text.setText(text)

        if level.value is None:

            self.msg_icon.setVisible(False)

        else:

            icon = QApplication.style().standardIcon(level.value)
            self.msg_icon.setPixmap(icon.pixmap(16, 16))
            self.msg_icon.setVisible(True)

    # %% Button handlers ----

    def _on_prev_clicked(self):

        self._save_current_annotations()
        self._load_image(self.idx - 1)

    def _on_next_clicked(self):

        self._save_current_annotations()
        self._load_image(self.idx + 1)

    def _on_save_clicked(self):

        self._save_current_annotations()

        out_path = self.out_dir_path / "imgs_annotated.pkl"

        with open(out_path, "wb") as f:
            pickle.dump(self.imgs, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._unsaved_changes = False

        self.set_message(f"Annotations saved to:\n{out_path}", MessageLevel.SAVE)


# %% validate_annotation_inputs() ----


def validate_annotation_inputs(in_dir_path: str, out_dir_path: str) -> None:
    """Raise ValueError with a clear message if the annotation inputs are unusable."""

    in_dir = Path(in_dir_path).expanduser()
    out_dir = Path(out_dir_path).expanduser()

    annotated_pkl = out_dir / "imgs_annotated.pkl"
    source_pkl = in_dir / "imgs.pkl"

    if annotated_pkl.exists():
        pkl_path = annotated_pkl
    elif source_pkl.exists():
        pkl_path = source_pkl
    else:
        raise ValueError(
            f"No input file found.\n"
            f"Expected one of:\n"
            f"  {annotated_pkl}\n"
            f"  {source_pkl}"
        )

    try:
        with open(pkl_path, "rb") as f:
            imgs = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Could not load {pkl_path}:\n{type(e).__name__}: {e}") from e

    if not isinstance(imgs, dict) or len(imgs) == 0:
        raise ValueError(
            f"{pkl_path} must be a non-empty dict, got {type(imgs).__name__}."
        )

    for fn, entry in imgs.items():

        if not isinstance(entry, dict) or "data" not in entry:
            raise ValueError(f"{pkl_path}: entry '{fn}' is missing the 'data' key.")

        ch_data = entry["data"]

        if not isinstance(ch_data, dict) or len(ch_data) == 0:
            raise ValueError(
                f"{pkl_path}: 'data' for entry '{fn}' must be a non-empty dict."
            )

        for ch, ch_entry in ch_data.items():

            for key in ("norm_img", "img"):

                if key not in ch_entry:
                    raise ValueError(
                        f"{pkl_path}: entry '{fn}', channel '{ch}' is missing '{key}'."
                    )

                if not isinstance(ch_entry[key], np.ndarray):
                    raise ValueError(
                        f"{pkl_path}: entry '{fn}', channel '{ch}': "
                        f"'{key}' must be a numpy array, got {type(ch_entry[key]).__name__}."
                    )


# %% launch_annotation_viewer() ----


def launch_annotation_viewer(in_dir_path: str, out_dir_path: str):

    import napari

    viewer = napari.Viewer()

    widget = AnnotationManager(
        viewer=viewer,
        in_dir_path=in_dir_path,
        out_dir_path=out_dir_path,
    )

    viewer.window.add_dock_widget(widget, area="right", name="Annotation")

    napari.run()
