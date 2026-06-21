# %% Import required libraries ----

import json

from qtpy.QtGui import QFont

from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QTabWidget,
    QScrollArea,
    QLabel,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
)

from . import configuration


# %% Small widget helpers ----


def _pretty_population(key: str) -> str:
    """Display form of a population key: '-pos'/'-neg' -> superscript +/-."""

    return key.replace("-pos", "⁺").replace("-neg", "⁻")


def _percent_spin(value: float) -> QDoubleSpinBox:
    """A 0-100 percentile spin box (4 decimals covers values like 99.9999)."""

    spin = QDoubleSpinBox()
    spin.setRange(0.0, 100.0)
    spin.setDecimals(4)
    spin.setValue(float(value))

    return spin


def _combo(options: list, current: str) -> QComboBox:
    """A combo box pre-filled with 'options' and set to 'current'."""

    combo = QComboBox()
    combo.addItems(options)
    combo.setCurrentText(current)

    return combo


# %% TubeIdSetWidget() ----


class TubeIdSetWidget(QGroupBox):
    """One tube_id_matching set: a 'match' string + one tube combo per population."""

    def __init__(self, population_keys: list, tube_options: list, set_data: dict):

        super().__init__()

        layout = QVBoxLayout(self)

        # Match string + remove button on the top row
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("match"))
        self.edit_match = QLineEdit(set_data.get("match", ""))
        top_row.addWidget(self.edit_match, stretch=1)
        self.btn_remove = QPushButton("Remove")
        top_row.addWidget(self.btn_remove)
        layout.addLayout(top_row)

        # One tube combo per population
        form = QFormLayout()
        self._tube_combos = {}
        tube_ids = set_data.get("tube_ids", {})
        for key in population_keys:
            combo = _combo(tube_options, tube_ids.get(key, tube_options[0]))
            self._tube_combos[key] = combo
            form.addRow(_pretty_population(key), combo)
        layout.addLayout(form)

    def value(self) -> dict:

        return {
            "match": self.edit_match.text(),
            "tube_ids": {k: c.currentText() for k, c in self._tube_combos.items()},
        }


# %% ConfigEditor() ----


class ConfigEditor(QDialog):
    """Standalone dialog to edit the napari-bruce configuration."""

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("napari-bruce configuration")
        self.setMinimumSize(520, 640)

        self.config = configuration.get_config()

        # Channel keys (int in-memory) and the 5 population keys.
        self._channel_keys = sorted(self.config["channels"].keys())
        self._population_keys = [
            k for k in self.config["elements"].keys() if k != "tube_id_matching"
        ]

        # Option lists used to seed combos.
        self._model_options = list(configuration.list_stardist_models().keys())
        self._channel_colors = configuration.list_channel_colors()
        self._population_colors = configuration.list_population_colors()
        self._laser_options = configuration.list_laser_functions()
        self._tube_options = configuration.list_tube_ids()

        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_paths_tab(), "Paths")
        tabs.addTab(self._build_channels_tab(), "Channels")
        tabs.addTab(self._build_elements_tab(), "Elements")
        tabs.addTab(self._build_annotation_tab(), "Annotation")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Enlarge group-box titles consistently across all tabs.
        self._style_group_titles()

    # -- Tab builders -----------------------------------------------------

    @staticmethod
    def _scrollable(inner: QWidget) -> QScrollArea:

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        return scroll

    def _style_group_titles(self) -> None:
        """Enlarge/bold every group-box title while leaving contents at the
        default font, so titles look the same across all tabs."""

        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)

        # Explicit base font (point size + weight set so its resolve mask is
        # non-empty). A plain QApplication.font() has an empty resolve mask, so
        # assigning it would NOT override the inherited enlarged title font and the
        # contents would be enlarged too.
        base_font = QFont()
        base_font.setPointSize(QApplication.font().pointSize())
        base_font.setBold(False)

        for box in self.findChildren(QGroupBox):
            box.setFont(title_font)
            for child in box.findChildren(QWidget):
                child.setFont(base_font)

    def _build_paths_tab(self) -> QWidget:

        page = QWidget()
        page_layout = QVBoxLayout(page)

        # Input / output directories with Browse buttons
        paths_box = QGroupBox("Directories")
        paths_form = QFormLayout(paths_box)
        self.edit_in_dir = QLineEdit(self.config["in_dir_path"])
        self.edit_out_dir = QLineEdit(self.config["out_dir_path"])
        paths_form.addRow("input directory", self._with_browse(self.edit_in_dir))
        paths_form.addRow("output directory", self._with_browse(self.edit_out_dir))
        page_layout.addWidget(paths_box)

        page_layout.addStretch()

        return self._scrollable(page)

    def _build_channels_tab(self) -> QWidget:

        page = QWidget()
        page_layout = QVBoxLayout(page)

        # Per-channel detection settings
        self._channel_widgets = {}
        for key in self._channel_keys:
            ch = self.config["channels"][key]
            box = QGroupBox(f"Channel {key}")
            form = QFormLayout(box)
            w = {
                "name": QLineEdit(ch["name"]),
                "low_pct": _percent_spin(ch["low_pct"]),
                "high_pct": _percent_spin(ch["high_pct"]),
                "stardist_model": _combo(self._model_options, ch["stardist_model"]),
                "min_area_um2": self._area_spin(ch["min_area_um2"]),
                "color": _combo(self._channel_colors, ch["color"]),
            }
            form.addRow("name", w["name"])
            form.addRow("low percentile", w["low_pct"])
            form.addRow("high percentile", w["high_pct"])
            form.addRow("StarDist model", w["stardist_model"])
            form.addRow("min area (µm²)", w["min_area_um2"])
            form.addRow("color", w["color"])
            self._channel_widgets[key] = w
            page_layout.addWidget(box)

        # Overlap thresholds
        ovl_box = QGroupBox("Overlap thresholds")
        ovl_form = QFormLayout(ovl_box)
        self.spin_ovl_0by1 = _percent_spin(self.config["min_pct_ovl_ch0_by_ch1"])
        self.spin_ovl_1by0 = _percent_spin(self.config["min_pct_ovl_ch1_by_ch0"])
        ovl_form.addRow("min % overlap ch0 by ch1", self.spin_ovl_0by1)
        ovl_form.addRow("min % overlap ch1 by ch0", self.spin_ovl_1by0)
        page_layout.addWidget(ovl_box)

        page_layout.addStretch()

        return self._scrollable(page)

    def _build_elements_tab(self) -> QWidget:

        page = QWidget()
        page_layout = QVBoxLayout(page)

        # The 5 cross-channel populations
        self._element_widgets = {}
        for key in self._population_keys:
            elem = self.config["elements"][key]
            box = QGroupBox(_pretty_population(key))
            form = QFormLayout(box)
            w = {
                "color": _combo(self._population_colors, elem["color"]),
                "collect": QCheckBox(),
                "laser_function": _combo(
                    self._laser_options, elem["laser_function"]
                ),
                "tube_id": _combo(self._tube_options, elem["tube_id"]),
            }
            w["collect"].setChecked(bool(elem["collect"]))
            form.addRow("color", w["color"])
            form.addRow("collect", w["collect"])
            form.addRow("laser function", w["laser_function"])
            form.addRow("default tube ID", w["tube_id"])
            self._element_widgets[key] = w
            page_layout.addWidget(box)

        # tube_id_matching: enable flag + dynamic list of sets
        matching = self.config["elements"]["tube_id_matching"]
        match_box = QGroupBox("Tube ID filename matching")
        match_layout = QVBoxLayout(match_box)

        self.check_matching_enabled = QCheckBox("Enable")
        self.check_matching_enabled.setChecked(bool(matching["enabled"]))
        match_layout.addWidget(self.check_matching_enabled)

        self._sets_layout = QVBoxLayout()
        match_layout.addLayout(self._sets_layout)

        self._set_widgets = []
        for set_data in matching["sets"]:
            self._add_set_widget(set_data)

        btn_add = QPushButton("Add set")
        btn_add.clicked.connect(lambda: self._add_set_widget({}))
        match_layout.addWidget(btn_add)

        page_layout.addWidget(match_box)
        page_layout.addStretch()

        return self._scrollable(page)

    def _build_annotation_tab(self) -> QWidget:

        page = QWidget()
        page_layout = QVBoxLayout(page)

        self._annotation_widgets = {}
        for key in sorted(self.config["channels_annotation"].keys()):
            ann = self.config["channels_annotation"][key]
            box = QGroupBox(f"Channel {key}")
            form = QFormLayout(box)
            w = {
                "low_pct": _percent_spin(ann["low_pct"]),
                "high_pct": _percent_spin(ann["high_pct"]),
                "color": _combo(self._channel_colors, ann["color"]),
            }
            form.addRow("low percentile", w["low_pct"])
            form.addRow("high percentile", w["high_pct"])
            form.addRow("color", w["color"])
            self._annotation_widgets[key] = w
            page_layout.addWidget(box)

        page_layout.addStretch()

        return self._scrollable(page)

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _area_spin(value: float) -> QDoubleSpinBox:

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1_000_000.0)
        spin.setDecimals(1)
        spin.setValue(float(value))

        return spin

    def _with_browse(self, line_edit: QLineEdit) -> QWidget:
        """Wrap a path line edit with a 'Browse…' directory picker."""

        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, stretch=1)
        btn = QPushButton("Browse…")

        def _pick():
            path = QFileDialog.getExistingDirectory(
                self, "Select directory", line_edit.text()
            )
            if path:
                line_edit.setText(path)

        btn.clicked.connect(_pick)
        row.addWidget(btn)

        return wrapper

    def _add_set_widget(self, set_data: dict) -> None:

        widget = TubeIdSetWidget(
            population_keys=self._population_keys,
            tube_options=self._tube_options,
            set_data=set_data,
        )
        widget.btn_remove.clicked.connect(lambda: self._remove_set_widget(widget))
        self._set_widgets.append(widget)
        self._sets_layout.addWidget(widget)

    def _remove_set_widget(self, widget: TubeIdSetWidget) -> None:

        self._set_widgets.remove(widget)
        self._sets_layout.removeWidget(widget)
        widget.deleteLater()

    # -- Save -------------------------------------------------------------

    def _collect_config(self) -> dict:
        """Rebuild the config dict from the widgets, in canonical key order.

        Channel keys are written as strings ("0"/"1") to match the on-disk JSON
        and the keys check_config_integrity expects.
        """

        channels = {}
        for key in self._channel_keys:
            w = self._channel_widgets[key]
            channels[str(key)] = {
                "name": w["name"].text(),
                "low_pct": w["low_pct"].value(),
                "high_pct": w["high_pct"].value(),
                "stardist_model": w["stardist_model"].currentText(),
                "min_area_um2": w["min_area_um2"].value(),
                "color": w["color"].currentText(),
            }

        elements = {}
        for key in self._population_keys:
            w = self._element_widgets[key]
            elements[key] = {
                "color": w["color"].currentText(),
                "collect": w["collect"].isChecked(),
                "laser_function": w["laser_function"].currentText(),
                "tube_id": w["tube_id"].currentText(),
            }
        elements["tube_id_matching"] = {
            "enabled": self.check_matching_enabled.isChecked(),
            "sets": [w.value() for w in self._set_widgets],
        }

        channels_annotation = {}
        for key in sorted(self.config["channels_annotation"].keys()):
            w = self._annotation_widgets[key]
            channels_annotation[str(key)] = {
                "low_pct": w["low_pct"].value(),
                "high_pct": w["high_pct"].value(),
                "color": w["color"].currentText(),
            }

        return {
            "in_dir_path": self.edit_in_dir.text(),
            "out_dir_path": self.edit_out_dir.text(),
            "channels": channels,
            "min_pct_ovl_ch0_by_ch1": self.spin_ovl_0by1.value(),
            "min_pct_ovl_ch1_by_ch0": self.spin_ovl_1by0.value(),
            "elements": elements,
            "channels_annotation": channels_annotation,
        }

    def _on_save(self) -> None:

        config = self._collect_config()

        try:

            configuration.check_config_integrity(config)

        except configuration.ConfigError as e:

            QMessageBox.critical(self, "Invalid configuration", str(e))

            return

        with open(configuration.get_config_file_path(), "w") as f:

            json.dump(config, f, indent=2)

        self.accept()


# %% launch_config_dialog() ----


def launch_config_dialog() -> None:
    """Open the standalone configuration editor window."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = ConfigEditor()
    result = dialog.exec_()

    if result == QDialog.Accepted:
        print(f"Configuration saved at:\n{configuration.get_config_file_path()}")
    else:
        print("Configuration editing cancelled; no changes saved.")
