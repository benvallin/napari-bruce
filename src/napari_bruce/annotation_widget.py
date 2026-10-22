# %% Set up ----

import os
import pickle
import traceback
from pathlib import Path

import numpy as np
from enum import Enum
from qtpy.QtCore import QEvent, QObject, QThread, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
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
from .main_widget import ParamValueBox, make_color_overlay_delegate

# Get channel-specific ROI colors and default normalization settings from configuration
config = configuration.get_config()
CHANNEL_COLORS = [
    config["channels_annotation"][i]["color"]
    for i in sorted(config["channels_annotation"].keys())
]
NORM_SETTING_KEYS = [
    "background_subtraction",
    "radius",
    "robust_normalization",
    "low_pct",
    "high_pct",
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


# %% SavePklWorker ----


class SavePklWorker(QObject):

    sig_success = Signal(object)
    sig_error = Signal(str)

    def __init__(self, imgs: dict, out_path: Path, parent=None):

        super().__init__(parent)
        self.imgs = imgs
        self.out_path = out_path

    def run(self):

        # Atomic save: write to a temp file in the same directory, flush it to
        # disk, then replace the target in one step. An interrupted write thus
        # leaves the previous imgs_annotated.pkl intact and never produces a
        # truncated file.
        tmp_path = self.out_path.with_name(self.out_path.name + ".tmp")

        try:

            with open(tmp_path, "wb") as f:
                pickle.dump(self.imgs, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self.out_path)

            self.sig_success.emit(self.out_path)

        except Exception as e:

            # Clean up the partial temp file; never touch the real target.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

            self.sig_error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# %% CloseEventFilter ----


class CloseEventFilter(QObject):

    def __init__(self, owner: "AnnotationManager", parent=None):

        super().__init__(parent)
        self._owner = owner

    def eventFilter(self, obj, event):

        if event.type() == QEvent.Close:

            # A save is already in flight — hold the close until it finishes.
            if self._owner._saving:

                self._owner._close_after_save = True
                event.ignore()
                return True

            if self._owner._unsaved_changes:

                reply = QMessageBox.question(
                    self._owner.viewer.window._qt_window,
                    "Unsaved annotations",
                    "You have unsaved annotations.\nSave before closing?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                    QMessageBox.Save,
                )

                if reply == QMessageBox.Save:

                    # Defer the close until the async save has fully written
                    # the data; _on_save_finished closes the window.
                    self._owner._close_after_save = True
                    self._owner._on_save_clicked()
                    event.ignore()
                    return True

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

        self.img_layers = []
        self.shape_layers = []

        # Live normalization settings per channel index. Reset to the config
        # defaults every time an image is loaded; edited via the settings block
        # and applied to the displayed image by "Reprocess images".
        self._norm_settings = {}
        self._settings_widgets = {}

        self._unsaved_changes = False
        self._loading = False
        self._saving = False
        self._close_after_save = False

        self._close_filter = CloseEventFilter(owner=self)
        self.viewer.window._qt_window.installEventFilter(self._close_filter)

        self._build_layout()
        self._install_layer_colors()
        self._install_layer_delete_guard()
        self._remove_layer_buttons()
        self._start_load_thread()

    def closeEvent(self, event):

        t = getattr(self, "_load_thread", None)

        if isinstance(t, QThread) and t.isRunning():

            t.quit()
            t.wait()

        # Let any in-progress save finish writing before the widget is torn
        # down (the atomic write protects the file even if this is skipped).
        s = getattr(self, "_save_thread", None)

        try:

            if isinstance(s, QThread) and s.isRunning():

                s.wait()

        except RuntimeError:

            pass

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

        self.btn_reprocess = QPushButton("Reprocess images")
        self.btn_reprocess.clicked.connect(self._on_reprocess_clicked)
        self.btn_reprocess.setEnabled(False)

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
        self._add_norm_settings_boxes(layout)
        layout.addWidget(self.btn_reprocess)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.msg_container)

        self.setLayout(layout)
        self.setMinimumWidth(300)
        self.setMaximumWidth(1000)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _add_norm_settings_boxes(self, layout: QVBoxLayout):
        """Add the per-channel background-subtraction / robust-normalization controls.

        Mirrors the main widget: for each channel, a bold colored header, the
        background-subtraction checkbox + radius box, the robust-normalization
        checkbox, then the low/high clip-pct boxes.
        """

        for i in sorted(config["channels_annotation"].keys()):

            ch_config = config["channels_annotation"][i]

            lbl_header = QLabel(f"Channel {i}")
            lbl_header.setStyleSheet(f"font-weight: bold; color: {ch_config['color']};")

            chk_bg_sub = QCheckBox("Background subtraction")
            chk_bg_sub.toggled.connect(
                lambda checked, ch=i: self._on_norm_setting_changed(
                    ch, "background_subtraction", bool(checked)
                )
            )

            box_radius = ParamValueBox(
                label="radius (px)", default=ch_config["radius"], min_val=1, max_val=1000
            )
            box_radius.valueChanged.connect(
                lambda x, ch=i: self._on_norm_setting_changed(ch, "radius", float(x))
            )

            chk_robust_norm = QCheckBox("Robust normalization")
            chk_robust_norm.toggled.connect(
                lambda checked, ch=i: self._on_norm_setting_changed(
                    ch, "robust_normalization", bool(checked)
                )
            )

            box_low_pct = ParamValueBox(
                label="low % clip",
                default=ch_config["low_pct"],
                min_val=0.0,
                max_val=100.0,
                decimals=4,
            )
            box_low_pct.valueChanged.connect(
                lambda x, ch=i: self._on_norm_setting_changed(ch, "low_pct", float(x))
            )

            box_high_pct = ParamValueBox(
                label="high % clip",
                default=ch_config["high_pct"],
                min_val=0.0,
                max_val=100.0,
                decimals=4,
            )
            box_high_pct.valueChanged.connect(
                lambda x, ch=i: self._on_norm_setting_changed(ch, "high_pct", float(x))
            )

            widgets = [lbl_header, chk_bg_sub, box_radius, chk_robust_norm, box_low_pct, box_high_pct]

            for w in widgets:
                layout.addWidget(w)

            self._settings_widgets[i] = {
                "header": lbl_header,
                "background_subtraction": chk_bg_sub,
                "radius": box_radius,
                "robust_normalization": chk_robust_norm,
                "low_pct": box_low_pct,
                "high_pct": box_high_pct,
                "all": widgets,
            }

            self._norm_settings[i] = {}

        self._reset_norm_settings()

    def _on_norm_setting_changed(self, ch_idx: int, key: str, value):

        self._norm_settings[ch_idx][key] = value

        # Clip-pct only has an effect when robust normalization is on.
        if key == "robust_normalization":

            w = self._settings_widgets[ch_idx]
            w["low_pct"].setEnabled(value)
            w["high_pct"].setEnabled(value)

    def _reset_norm_settings(self):
        """Reapply the config defaults to the live settings and the controls."""

        for i, w in self._settings_widgets.items():

            defaults = config["channels_annotation"][i]

            for key in NORM_SETTING_KEYS:

                self._norm_settings[i][key] = defaults[key]

            w["background_subtraction"].setChecked(defaults["background_subtraction"])
            w["radius"].setValue(defaults["radius"])
            w["robust_normalization"].setChecked(defaults["robust_normalization"])
            w["low_pct"].setValue(defaults["low_pct"])
            w["high_pct"].setValue(defaults["high_pct"])
            w["low_pct"].setEnabled(defaults["robust_normalization"])
            w["high_pct"].setEnabled(defaults["robust_normalization"])

    def _normalize_channel(self, ch_idx: int, data: dict) -> np.ndarray:
        """Normalized display image for a channel, computed from the raw image
        under the live settings."""

        s = self._norm_settings[ch_idx]

        return workflow.normalize_img(
            img=data["img"],
            background_subtraction=s["background_subtraction"],
            radius=s["radius"],
            use_robust_normalization=s["robust_normalization"],
            low_pct=s["low_pct"],
            high_pct=s["high_pct"],
        )

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

        self._load_image(0)

        self.btn_reprocess.setEnabled(True)
        self.btn_save.setEnabled(True)

    def _on_load_error(self, error_msg: str):

        print(f"Failed to load imgs.pkl:\n\n{error_msg}")

    # %% Viewer layers ----

    # Layer-control properties the user can change via napari's "layer controls"
    # panel. Snapshotted when the layers are created and restored on every image
    # load so each image starts from identical defaults. (Image contrast_limits
    # is excluded: it is re-fitted from the data on load. Shapes edge_width is
    # per-shape, so only the "current" value is restored.)
    _IMAGE_CONTROLS = ("opacity", "gamma", "colormap", "blending", "interpolation2d")
    _SHAPES_CONTROLS = (
        "opacity",
        "blending",
        "current_edge_width",
        "current_edge_color",
        "current_face_color",
    )

    def _iter_control_specs(self):
        for layer in self.img_layers:
            yield layer, self._IMAGE_CONTROLS
        for layer in self.shape_layers:
            yield layer, self._SHAPES_CONTROLS

    def _snapshot_layer_controls(self):
        self._layer_control_defaults = []
        for layer, names in self._iter_control_specs():
            snap = {}
            for name in names:
                if not hasattr(layer, name):
                    continue
                value = getattr(layer, name)
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
                    # Tolerate napari version differences in control setters;
                    # a single failed control must not abort the image load.
                    pass

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

        DelegateClass = make_color_overlay_delegate(type(qt_list.itemDelegate()))
        self._qt_layer_list = qt_list
        self._layer_delegate = DelegateClass(self._build_layer_colors())
        qt_list.setItemDelegate(self._layer_delegate)
        if hasattr(self._layer_delegate, "loading_frame_changed"):
            self._layer_delegate.loading_frame_changed.connect(
                qt_list.viewport().update
            )

    def _build_layer_colors(self):
        layer_colors = {}
        for i, (img_layer, shape_layer) in enumerate(
            zip(self.img_layers, self.shape_layers)
        ):
            qcolor = QColor(CHANNEL_COLORS[i % len(CHANNEL_COLORS)])
            qcolor.setAlpha(30)
            layer_colors[img_layer.name] = qcolor
            layer_colors[shape_layer.name] = qcolor
        return layer_colors

    def _refresh_layer_colors(self):
        if hasattr(self, "_layer_delegate"):
            self._layer_delegate._layer_colors = self._build_layer_colors()
            self._qt_layer_list.viewport().update()

    def _install_layer_delete_guard(self):
        # Prevent the user from accidentally deleting the managed layers. napari
        # routes every layer-list deletion (Delete/Backspace key and the
        # viewer's trash button) through LayerList.remove_selected(), so we wrap
        # it to veto removal whenever a managed layer is selected. The managed
        # set is read at call time because the pool is rebuilt when the channel
        # count changes.
        layers = self.viewer.layers
        original_remove_selected = layers.remove_selected

        def guarded_remove_selected():
            if set(self.img_layers + self.shape_layers) & set(layers.selection):
                self.viewer.status = "These layers can't be deleted."
                return
            original_remove_selected()

        layers.remove_selected = guarded_remove_selected

    def _remove_layer_buttons(self):
        # Hide the new-layer and delete buttons above the layer list: the
        # managed layers are fixed, so the user should never add or delete one.
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

    def _on_shapes_changed(self, *args, **kwargs):

        if not self._loading:

            self._unsaved_changes = True

    def _rebuild_viewer_layers(self, ch_names_data: list):
        """Replace all managed layers with a fresh set sized to ch_names_data."""

        for layer in self.img_layers + self.shape_layers:
            self.viewer.layers.remove(layer)

        self.img_layers = []
        self.shape_layers = []

        # Image layers added first — appear below shapes in the layer list
        for cn in ch_names_data:
            self.img_layers.append(
                self.viewer.add_image(
                    np.zeros((1, 1), dtype=np.uint8),
                    name=f"\u25ce {cn} image",
                    visible=True,
                )
            )

        # Shapes layers added second — appear above images in the layer list
        for i, cn in enumerate(ch_names_data):
            col = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            layer = self.viewer.add_shapes(
                data=[],
                name=f"\u25b6 add {cn}",
                edge_color=col,
                face_color="transparent",
                edge_width=2,
                visible=True,
            )
            layer.events.data.connect(self._on_shapes_changed)
            self.shape_layers.append(layer)

        # Capture the freshly created layer-control values so that loading
        # another image can restore them, undoing any change the user made in
        # napari's "layer controls" panel.
        self._snapshot_layer_controls()

    def _load_image(self, idx: int):

        fn = self.img_keys[idx]
        fn_data = self.imgs[fn]["data"]
        ch_names_data = list(fn_data.keys())
        n_ch = len(ch_names_data)

        self._loading = True

        # Settings changed on the previous image do not carry over
        self._reset_norm_settings()

        for i, w in self._settings_widgets.items():

            for widget in w["all"]:
                widget.setVisible(i < n_ch)

            if i < n_ch:
                w["header"].setText(f"Channel {i}: {ch_names_data[i]}")

        with self.viewer.layers.events.blocker():

            if len(self.img_layers) != n_ch:
                # Channel count changed — rebuild the pool (rare)
                self._rebuild_viewer_layers(ch_names_data)

            else:
                # User changes in napari's layer controls do not carry over
                # to the next image
                self._restore_layer_controls()

                # Same channel count — rename to placeholders first to avoid
                # conflicts when a channel name matches a slot from the previous image
                for i in range(n_ch):
                    self.img_layers[i].name = f"__slot{i}__"
                    self.shape_layers[i].name = f"__slot{i}_rois__"

            for i, cn in enumerate(ch_names_data):

                data = fn_data[cn]
                col = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]

                self.img_layers[i].data = self._normalize_channel(i, data)
                self.img_layers[i].name = f"\u25ce {cn} image"
                self.img_layers[i].reset_contrast_limits()
                self.img_layers[i].visible = True

                existing_rois = data.get("rois", [])
                layer = self.shape_layers[i]
                layer.data = (
                    [arr.copy() for arr in existing_rois] if existing_rois else []
                )

                if existing_rois:
                    layer.selected_data = set(range(len(existing_rois)))
                    layer.edge_width = 2
                    layer.selected_data = set()

                layer.name = f"\u25b6 add {cn}"
                layer.current_face_color = "transparent"
                layer.current_edge_color = col
                layer.current_edge_width = 2
                layer.visible = True

        # Set mode after the blocker exits so the viewer always fires a mode
        # event and updates its toolbar — setting it inside the blocker suppresses
        # the event, leaving the toolbar stale on rebuild and the next in-place load
        for layer in self.shape_layers:
            layer.mode = "add_path"

        self._loading = False

        self._refresh_layer_colors()

        self.viewer.reset_view()

        self.idx = idx
        self._update_counter()
        self._update_nav_buttons()

    # %% Annotation save / convert ----

    def _save_current_annotations(self):

        fn = self.img_keys[self.idx]
        fn_data = self.imgs[fn]["data"]

        # Record the preprocessing settings the image was annotated under, by
        # channel name, so a model trained on these annotations can reproduce
        # the same normalization from the raw img.
        self.imgs[fn]["channels_annotation"] = {
            cn: {k: self._norm_settings[i][k] for k in NORM_SETTING_KEYS}
            for i, cn in enumerate(fn_data.keys())
        }

        for i, cn in enumerate(fn_data.keys()):

            layer = self.shape_layers[i]
            rois = [arr.copy() for arr in layer.data]
            ch = fn_data[cn]

            ch["rois"] = rois

            empty_msk = np.zeros(ch["img"].shape, dtype=np.uint16)

            if rois:

                msk = workflow.append_shapes_to_msk(
                    msk=empty_msk,
                    shapes=rois,
                    start_idx=1,
                )
                ch["msk"] = msk.astype(np.uint16)

            else:

                ch["msk"] = empty_msk

    # %% Message ----

    def _set_message(self, text: str, level: MessageLevel = MessageLevel.NONE):

        self.msg_text.setText(text)

        if level.value is None:

            self.msg_icon.setVisible(False)

        else:

            icon = QApplication.style().standardIcon(level.value)
            self.msg_icon.setPixmap(icon.pixmap(16, 16))
            self.msg_icon.setVisible(True)

    # %% Button handlers ----

    def _on_reprocess_clicked(self):
        """Recompute the displayed images of the current image with the live settings."""

        fn_data = self.imgs[self.img_keys[self.idx]]["data"]

        for i, cn in enumerate(fn_data.keys()):

            self.img_layers[i].data = self._normalize_channel(i, fn_data[cn])
            self.img_layers[i].reset_contrast_limits()

    def _on_prev_clicked(self):

        self._save_current_annotations()
        self._load_image(self.idx - 1)

    def _on_next_clicked(self):

        self._save_current_annotations()
        self._load_image(self.idx + 1)

    def _on_save_clicked(self):

        # Ignore re-entrant clicks while a save is already running.
        if self._saving:

            return

        self._save_current_annotations()

        out_path = self.out_dir_path / "imgs_annotated.pkl"

        # Run the (potentially long) pickle write on a worker thread so the
        # window stays responsive and cannot be force-closed as "Not
        # Responding" mid-save.
        self._saving = True
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._set_message(
            "Saving annotations — please do not close the window…",
            MessageLevel.BUSY,
        )

        self._save_thread = QThread()
        self._save_worker = SavePklWorker(imgs=self.imgs, out_path=out_path)

        self._save_worker.moveToThread(self._save_thread)
        self._save_thread.started.connect(self._save_worker.run)
        self._save_worker.sig_success.connect(self._on_save_finished)
        self._save_worker.sig_error.connect(self._on_save_error)
        self._save_worker.sig_success.connect(self._save_thread.quit)
        self._save_worker.sig_error.connect(self._save_thread.quit)
        self._save_thread.finished.connect(self._save_worker.deleteLater)
        self._save_thread.finished.connect(self._save_thread.deleteLater)

        self._save_thread.start()

    def _restore_buttons_after_save(self):

        self._saving = False
        self.btn_prev.setEnabled(self.idx > 0)
        self.btn_next.setEnabled(self.idx < self.n_imgs - 1)
        self.btn_save.setEnabled(True)

    def _on_save_finished(self, out_path: Path):

        self._unsaved_changes = False
        self._restore_buttons_after_save()
        self._set_message(f"Annotations saved to:\n{out_path}", MessageLevel.SAVE)

        # If the save was triggered by a close request, close now that the
        # data is safely on disk.
        if self._close_after_save:

            self._close_after_save = False
            self.viewer.window._qt_window.close()

    def _on_save_error(self, error_msg: str):

        self._close_after_save = False
        self._restore_buttons_after_save()
        self._set_message(
            f"Failed to save annotations:\n\n{error_msg}", MessageLevel.ERROR
        )


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

        n_cfg_ch = len(config["channels_annotation"])

        if len(ch_data) > n_cfg_ch:
            raise ValueError(
                f"{pkl_path}: entry '{fn}' has {len(ch_data)} channels but "
                f"config['channels_annotation'] defines {n_cfg_ch}."
            )

        for ch, ch_entry in ch_data.items():

            if "img" not in ch_entry:
                raise ValueError(
                    f"{pkl_path}: entry '{fn}', channel '{ch}' is missing 'img'."
                )

            if not isinstance(ch_entry["img"], np.ndarray):
                raise ValueError(
                    f"{pkl_path}: entry '{fn}', channel '{ch}': "
                    f"'img' must be a numpy array, got {type(ch_entry['img']).__name__}."
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
