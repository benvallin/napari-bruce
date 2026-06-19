# %% Import required libraries ----

from pathlib import Path

from qtpy.QtGui import QFont

from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QScrollArea,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
)

from . import configuration
from . import workflow


# %% Helpers ----


def _list_mergeable_subfolders(parent: str | Path) -> list[Path]:
    """Immediate subfolders of 'parent' that hold an elem_list_collect.txt."""

    parent = Path(parent).expanduser()

    if not parent.is_dir():
        return []

    return sorted(
        p
        for p in parent.iterdir()
        if p.is_dir() and (p / "elem_list_collect.txt").exists()
    )


# %% MergeElemLists() ----


class MergeElemLists(QDialog):
    """Standalone dialog to combine element lists from several bruce runs."""

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("napari-bruce — merge element lists")
        self.setMinimumSize(560, 560)

        self.config = configuration.get_config()

        # Selected image folders (deduped, order preserved) and their checked state.
        self._folders: list[Path] = []
        self._checked: dict[str, bool] = {}

        # Folder selection of the last completed merge (set of resolved path strings),
        # used to disable Merge until the selection changes again.
        self._last_merged_selection = None

        # Explicit base font (point size + weight set so its resolve mask is
        # non-empty) used to keep group-box CONTENTS at the default size while only
        # the titles are enlarged. A plain QApplication.font() has an empty resolve
        # mask, so assigning it does not override the inherited enlarged title font.
        self._content_font = QFont()
        self._content_font.setPointSize(QApplication.font().pointSize())
        self._content_font.setBold(False)

        layout = QVBoxLayout(self)

        # Parent folder holding the per-image result subfolders.
        parent_box = QGroupBox("Results folder")
        parent_layout = QHBoxLayout(parent_box)
        self.edit_parent = QLineEdit(self.config["out_dir_path"])
        parent_layout.addWidget(self.edit_parent, stretch=1)
        btn_browse_parent = QPushButton("Browse…")
        btn_browse_parent.clicked.connect(self._on_browse_parent)
        parent_layout.addWidget(btn_browse_parent)
        layout.addWidget(parent_box)

        # Checkable list of image folders.
        folders_box = QGroupBox("Image folders to merge")
        folders_layout = QVBoxLayout(folders_box)

        select_row = QHBoxLayout()
        btn_add = QPushButton("Add folder…")
        btn_add.clicked.connect(self._on_add_folder)
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        select_row.addWidget(btn_add)
        select_row.addWidget(btn_all)
        select_row.addWidget(btn_none)
        select_row.addStretch()
        folders_layout.addLayout(select_row)

        self._folders_container = QWidget()
        self._folders_container_layout = QVBoxLayout(self._folders_container)
        self._folders_container_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._folders_container)
        folders_layout.addWidget(scroll, stretch=1)

        layout.addWidget(folders_box, stretch=1)

        # Output folder + file name for the merged list.
        out_box = QGroupBox("Save merged list to")
        out_layout = QVBoxLayout(out_box)
        dir_row = QHBoxLayout()
        self.edit_out = QLineEdit(self.config["out_dir_path"])
        dir_row.addWidget(self.edit_out, stretch=1)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._on_browse_out)
        dir_row.addWidget(btn_browse_out)
        out_layout.addLayout(dir_row)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("File name prefix"))
        self.edit_filename = QLineEdit("")
        name_row.addWidget(self.edit_filename, stretch=1)
        out_layout.addLayout(name_row)
        layout.addWidget(out_box)

        # Status message shown in-window after a merge (replaces a pop-up dialog).
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Action buttons.
        action_row = QHBoxLayout()
        action_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        action_row.addWidget(btn_close)
        self.btn_merge = QPushButton("Merge")
        self.btn_merge.setDefault(True)
        self.btn_merge.clicked.connect(self._on_merge)
        action_row.addWidget(self.btn_merge)
        layout.addLayout(action_row)

        # Seed the list from the default results folder (also sets Merge enabled state).
        self._add_folders(_list_mergeable_subfolders(self.edit_parent.text()))

        # Enlarge the group-box titles (Results folder, Image folders to merge, ...).
        self._style_group_titles()

    def _style_group_titles(self) -> None:
        """Enlarge/bold every group-box title while leaving contents at the default
        font, matching the --edit-config window."""

        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)

        for box in self.findChildren(QGroupBox):
            box.setFont(title_font)
            for child in box.findChildren(QWidget):
                child.setFont(self._content_font)

    # -- Folder list ------------------------------------------------------

    def _add_folders(self, paths) -> None:
        """Append new folders (deduped, default-checked) and rebuild the rows."""

        for p in paths:
            p = Path(p).expanduser().resolve()
            if p not in self._folders:
                self._folders.append(p)
                self._checked.setdefault(str(p), True)

        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Rebuild the folder rows from self._folders and their checked state."""

        while self._folders_container_layout.count():
            item = self._folders_container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self._folders:
            self._folders_container_layout.addWidget(
                QLabel("No image folders selected.")
            )
            self._update_merge_enabled()
            return

        for folder in self._folders:
            row = QWidget()
            # Rows are created after the group-box titles are enlarged; pin them to
            # the content font so new checkboxes don't inherit the big title font.
            row.setFont(self._content_font)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(folder.name)
            cb.setChecked(self._checked.get(str(folder), True))
            cb.setToolTip(str(folder))
            cb.toggled.connect(
                lambda checked, f=str(folder): self._on_checkbox_toggled(f, checked)
            )
            row_layout.addWidget(cb, stretch=1)

            btn_remove = QPushButton("Remove")
            btn_remove.clicked.connect(lambda _, f=folder: self._remove_folder(f))
            row_layout.addWidget(btn_remove)

            self._folders_container_layout.addWidget(row)

        self._folders_container_layout.addStretch()

        self._update_merge_enabled()

    def _remove_folder(self, folder: Path) -> None:

        if folder in self._folders:
            self._folders.remove(folder)
        self._checked.pop(str(folder), None)
        self._rebuild_rows()

    def _set_all_checked(self, checked: bool) -> None:

        for cb in self._checkboxes():
            cb.setChecked(checked)

        self._update_merge_enabled()

    def _on_checkbox_toggled(self, folder_key: str, checked: bool) -> None:

        self._checked[folder_key] = checked
        self._update_merge_enabled()

    def _update_merge_enabled(self) -> None:
        """Enable Merge only with >=2 folders selected and a selection that differs
        from the last completed merge."""

        selected = {str(f) for f in self._selected_folders()}
        unchanged = selected == self._last_merged_selection
        self.btn_merge.setEnabled(len(selected) >= 2 and not unchanged)

    def _checkboxes(self) -> list[QCheckBox]:

        return self._folders_container.findChildren(QCheckBox)

    def _selected_folders(self) -> list[Path]:

        return [f for f in self._folders if self._checked.get(str(f), False)]

    # -- Browse handlers --------------------------------------------------

    def _on_browse_parent(self) -> None:

        path = QFileDialog.getExistingDirectory(
            self, "Select results folder", self.edit_parent.text()
        )
        if path:
            self.edit_parent.setText(path)
            self.edit_out.setText(path)
            self._add_folders(_list_mergeable_subfolders(path))

    def _on_add_folder(self) -> None:

        path = QFileDialog.getExistingDirectory(
            self, "Add image folder", self.edit_parent.text()
        )
        if not path:
            return
        if not (Path(path) / "elem_list_collect.txt").exists():
            QMessageBox.warning(
                self,
                "No element list",
                f"This folder has no elem_list_collect.txt:\n{path}",
            )
            return
        self._add_folders([path])

    def _on_browse_out(self) -> None:

        path = QFileDialog.getExistingDirectory(
            self, "Select output folder", self.edit_out.text()
        )
        if path:
            self.edit_out.setText(path)

    # -- Merge ------------------------------------------------------------

    def _on_merge(self) -> None:

        # The Merge button is only enabled with >=2 selected folders and a changed
        # selection (see _update_merge_enabled), so no folder-count guard is needed.
        folders = self._selected_folders()

        out_dir = Path(self.edit_out.text()).expanduser()

        if not out_dir.is_dir():
            QMessageBox.critical(
                self, "Invalid output folder", f"Not a folder:\n{out_dir}"
            )
            return

        # The file name is "<prefix>_merged_elem_list<suffix>", where the optional
        # user prefix is prepended (omitted entirely when blank) and the suffix is
        # always the group name.
        prefix = self.edit_filename.text().strip()
        base = f"{prefix}_merged_elem_list" if prefix else "merged_elem_list"

        try:
            merged = workflow.merge_elem_lists(folder_paths=folders)
        except Exception as e:
            QMessageBox.critical(
                self, "Merge failed", f"{type(e).__name__}: {e}"
            )
            return

        # Always suffix the group name (collect / omit / all), mirroring single-run
        # output (elem_list_collect.txt, ...), so a collect-only merge still gets the
        # "_collect" suffix. To avoid overwriting a previous merge, bump a numeric
        # tag "(1)", "(2)", ... applied to the whole group so the set stays together
        # (collect/omit/all share one index) and no member collides with an existing
        # file.
        n = 0
        while True:
            tag = "" if n == 0 else f" ({n})"
            names = {nm: f"{base}_{nm}{tag}.txt" for nm in merged}
            if not any((out_dir / fn).exists() for fn in names.values()):
                break
            n += 1

        written = []
        for nm, txt in merged.items():
            path = out_dir / names[nm]
            with open(path, "w") as f:
                f.write(txt)
            written.append(path)

        paths = "\n".join(str(p) for p in written)
        print(f"Merged {len(folders)} element list(s) into:\n{paths}\n")

        # Report completion in-window rather than via a pop-up dialog.
        file_names = "\n".join(names[nm] for nm in merged)
        self.status_label.setText(
            f"✔ Merged {len(folders)} folder(s) into:\n{file_names}"
        )

        # Stay open so the user can run further merges without relaunching, but
        # disable Merge until the folder selection changes (avoids re-merging the
        # exact same set).
        self._last_merged_selection = {str(f) for f in folders}
        self._update_merge_enabled()


# %% launch_merge_elem_lists() ----


def launch_merge_elem_lists() -> None:
    """Open the standalone element-list merge window."""

    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication([])

    dialog = MergeElemLists()
    dialog.exec_()
