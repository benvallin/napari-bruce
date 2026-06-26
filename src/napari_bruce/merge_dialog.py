# %% Import required libraries ----

import sys
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
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


def _safe_print(text: str) -> None:
    """Print text, substituting superscript +/- when the console can't encode them."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.replace("⁺", "-pos").replace("⁻", "-neg")
        print(safe)


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


# %% OverlapResolutionDialog() ----


class OverlapResolutionDialog(QDialog):
    """Per-pair overlap review dialog shown before writing merged element lists.

    Displays one group box per overlapping pair.  Each pair shows the overlap
    statistics and one checkbox per element; unchecking an element marks it for
    exclusion from the merged output.  Elements that appear in more than one pair
    get a checkbox in each pair, kept in sync via blockSignals.
    """

    def __init__(self, overlaps: list[dict], parent=None):

        super().__init__(parent)
        self.setWindowTitle("Overlapping elements detected")
        self.setMinimumSize(560, 400)

        # keep[no] = True means the element is retained; False means excluded.
        self._keep: dict[int, bool] = {}
        # All QCheckBox widgets for each element no, for cross-pair syncing.
        self._cbs_by_no: dict[int, list[QCheckBox]] = {}

        # Collect all unique element nos upfront so _keep is fully populated
        # before any checkbox signal fires.
        for o in overlaps:
            for side in ("a", "b"):
                no = o[side]["no"]
                if no not in self._keep:
                    self._keep[no] = True
                    self._cbs_by_no[no] = []

        layout = QVBoxLayout(self)

        intro = QLabel(
            f"{len(overlaps)} overlapping element pair(s) detected across images.\n"
            "Uncheck element(s) to be excluded from the merged output."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)

        for i, o in enumerate(overlaps, start=1):

            a, b = o["a"], o["b"]

            box = QGroupBox(f"Pair {i}")
            box_layout = QVBoxLayout(box)

            pop_tag = "same population" if o["same_population"] else "different population"
            stats = QLabel(f"{o['inter_um2']:.0f} µm² overlap - {pop_tag}")
            box_layout.addWidget(stats)

            for side_key in ("a", "b"):
                e = o[side_key]
                no = e["no"]
                cb = QCheckBox(
                    f"#{no} • {e['img_nm']} | {e['label']} | {e['collect_str']}"
                )
                cb.setChecked(self._keep[no])
                cb.toggled.connect(
                    lambda checked, n=no: self._on_toggled(n, checked)
                )
                self._cbs_by_no[no].append(cb)
                box_layout.addWidget(cb)

            content_layout.addWidget(box)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_proceed = QPushButton("Proceed")
        btn_proceed.setDefault(True)
        btn_proceed.clicked.connect(self.accept)
        btn_row.addWidget(btn_proceed)
        layout.addLayout(btn_row)

        self._apply_title_font()

    def _on_toggled(self, no: int, checked: bool) -> None:
        self._keep[no] = checked
        # Sync every other checkbox for this element without retriggering signals.
        for cb in self._cbs_by_no.get(no, []):
            if cb.isChecked() != checked:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)

    def _apply_title_font(self) -> None:
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        content_font = QFont()
        content_font.setPointSize(QApplication.font().pointSize())
        content_font.setBold(False)
        for box in self.findChildren(QGroupBox):
            box.setFont(title_font)
            for child in box.findChildren(QWidget):
                child.setFont(content_font)

    def excluded_nos(self) -> set[int]:
        """Element nos the user chose to exclude (unchecked checkboxes)."""
        return {no for no, keep in self._keep.items() if not keep}


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

        # Resolved path strings of experiments added via "Add experiment". Refresh
        # keeps these as long as they still exist on disk, even when they live
        # outside the current input directory.
        self._manual: set[str] = set()

        # State of the last completed merge — (selected path strings, output dir,
        # file name prefix) — used to disable Merge until any of these changes again.
        self._last_merged_state = None

        # Explicit base font (point size + weight set so its resolve mask is
        # non-empty) used to keep group-box CONTENTS at the default size while only
        # the titles are enlarged. A plain QApplication.font() has an empty resolve
        # mask, so assigning it does not override the inherited enlarged title font.
        self._content_font = QFont()
        self._content_font.setPointSize(QApplication.font().pointSize())
        self._content_font.setBold(False)

        layout = QVBoxLayout(self)

        # Parent folder holding the per-image result subfolders.
        parent_box = QGroupBox("Input directory")
        parent_layout = QHBoxLayout(parent_box)
        self.edit_parent = QLineEdit(self.config["out_dir_path"])
        parent_layout.addWidget(self.edit_parent, stretch=1)
        btn_browse_parent = QPushButton("Browse…")
        btn_browse_parent.clicked.connect(self._on_browse_parent)
        parent_layout.addWidget(btn_browse_parent)
        layout.addWidget(parent_box)

        # Checkable list of image folders.
        folders_box = QGroupBox("Experiments to merge")
        folders_layout = QVBoxLayout(folders_box)

        select_row = QHBoxLayout()
        btn_add = QPushButton("Add experiment")
        btn_add.clicked.connect(self._on_add_folder)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._on_refresh)
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        select_row.addWidget(btn_add)
        select_row.addWidget(btn_refresh)
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

        # Output folder for the merged list.
        out_box = QGroupBox("Output directory")
        out_layout = QHBoxLayout(out_box)
        self.edit_out = QLineEdit(self.config["out_dir_path"])
        self.edit_out.textChanged.connect(self._update_merge_enabled)
        out_layout.addWidget(self.edit_out, stretch=1)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._on_browse_out)
        out_layout.addWidget(btn_browse_out)
        layout.addWidget(out_box)

        # File name prefix for the merged list.
        name_box = QGroupBox("File name")
        name_layout = QHBoxLayout(name_box)
        name_layout.addWidget(QLabel("Prefix"))
        self.edit_filename = QLineEdit("")
        self.edit_filename.textChanged.connect(self._update_merge_enabled)
        name_layout.addWidget(self.edit_filename, stretch=1)
        layout.addWidget(name_box)

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

        # Enlarge the group-box titles (Input directory, Experiments to merge, ...).
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
        """Append new folders (deduped, unchecked by default) and rebuild the rows."""

        for p in paths:
            p = Path(p).expanduser().resolve()
            if p not in self._folders:
                self._folders.append(p)
                self._checked.setdefault(str(p), False)

        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Rebuild the folder rows from self._folders and their checked state."""

        while self._folders_container_layout.count():
            item = self._folders_container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self._folders:
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
            cb.setChecked(self._checked.get(str(folder), False))
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
        self._manual.discard(str(folder))
        self._rebuild_rows()

    def _set_all_checked(self, checked: bool) -> None:

        for cb in self._checkboxes():
            cb.setChecked(checked)

        self._update_merge_enabled()

    def _on_checkbox_toggled(self, folder_key: str, checked: bool) -> None:

        self._checked[folder_key] = checked
        self._update_merge_enabled()

    def _update_merge_enabled(self) -> None:
        """Enable Merge only with >=2 folders selected and a state (selection,
        output directory, file name prefix) that differs from the last completed
        merge."""

        selected = {str(f) for f in self._selected_folders()}
        state = (selected, self.edit_out.text(), self.edit_filename.text())
        unchanged = state == self._last_merged_state
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

    def _on_refresh(self) -> None:
        """Resync the experiment list with the input directory.

        Mergeable subfolders of the input directory are added; current
        experiments that are no longer mergeable subfolders are dropped — except
        experiments added via "Add experiment", which are kept as long as their
        folder still holds an element list at its original location.
        """

        scanned = [
            Path(p).expanduser().resolve()
            for p in _list_mergeable_subfolders(self.edit_parent.text())
        ]
        scanned_keys = {str(p) for p in scanned}

        kept: list[Path] = []
        for folder in self._folders:
            key = str(folder)
            if key in scanned_keys:
                kept.append(folder)
            elif key in self._manual and (folder / "elem_list_collect.txt").exists():
                kept.append(folder)
            else:
                self._checked.pop(key, None)
                self._manual.discard(key)
        self._folders = kept

        # Add any newly available subfolders (deduped against what was kept).
        self._add_folders(scanned)

    def _on_add_folder(self) -> None:

        path = QFileDialog.getExistingDirectory(
            self, "Add experiment", self.edit_parent.text()
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
        self._manual.add(str(Path(path).expanduser().resolve()))
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
            merged, overlaps, collected, omitted = workflow.merge_elem_lists(folder_paths=folders)
        except Exception as e:
            QMessageBox.critical(
                self, "Merge failed", f"{type(e).__name__}: {e}"
            )
            return

        # Cut elements from different images whose contours overlap signal
        # partially overlapping FOVs → the same tissue could be captured twice.
        # Show the overlap resolution dialog so the user can inspect each pair
        # and choose which elements to keep before writing.
        excluded: set[int] = set()
        if overlaps:
            warning = workflow.format_overlap_warnings(overlaps)
            _safe_print(warning + "\n")
            resolution = OverlapResolutionDialog(overlaps, parent=self)
            if resolution.exec_() != QDialog.Accepted:
                self.status_label.setText(
                    f"✘ Merge cancelled: {len(overlaps)} overlapping element "
                    "pair(s) across images."
                )
                return
            excluded = resolution.excluded_nos()
            if excluded:
                try:
                    merged, _, collected, omitted = workflow.merge_elem_lists(
                        folder_paths=folders, exclude_nos=excluded
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self, "Merge failed", f"{type(e).__name__}: {e}"
                    )
                    return

        # Always suffix the group name (collect / omit / all), mirroring single-run
        # output (elem_list_collect.txt, ...), so a collect-only merge still gets the
        # "_collect" suffix. To avoid overwriting a previous merge, bump a numeric
        # tag "(1)", "(2)", ... applied to the whole group so the set stays together
        # (collect/omit/all/collection_summary share one index) and no member collides
        # with an existing file.
        summary_base = f"{prefix}_collection_summary" if prefix else "collection_summary"
        n = 0
        while True:
            tag = "" if n == 0 else f" ({n})"
            names = {nm: f"{base}_{nm}{tag}.txt" for nm in merged}
            summary_name = f"{summary_base}{tag}.txt"
            if (
                not any((out_dir / fn).exists() for fn in names.values())
                and not (out_dir / summary_name).exists()
            ):
                break
            n += 1

        written = []
        for nm, txt in merged.items():
            path = out_dir / names[nm]
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            written.append(path)

        summary_txt = workflow.make_merge_collection_summary(collected, omitted)
        summary_path = out_dir / summary_name
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_txt)
        written.append(summary_path)

        paths = "\n".join(str(p) for p in written)
        exclusion_note = (
            f"{len(excluded)} element(s) excluded to resolve "
            f"{len(overlaps)} overlapping pair(s).\n"
            if overlaps and excluded
            else ""
        )
        # Print collection header (COLLECTION SUMMARY: totals, no DETAILS) to terminal.
        collection_header = summary_txt.split("\nDETAILS:")[0].rstrip()
        _safe_print(f"Merged {len(folders)} element list(s) into:\n{paths}\n{exclusion_note}\n{collection_header}")

        # Report completion in-window rather than via a pop-up dialog. If the user
        # confirmed past an overlap warning, note that it was written anyway.
        file_names = "\n".join(names[nm] for nm in merged)
        if overlaps and excluded:
            overlap_note = (
                f"\n\n{len(excluded)} element(s) excluded to resolve "
                f"{len(overlaps)} overlapping pair(s)."
            )
        elif overlaps:
            overlap_note = (
                f"\n\n⚠ {len(overlaps)} overlapping pair(s) detected; "
                "all elements kept."
            )
        else:
            overlap_note = ""
        self.status_label.setText(
            f"✔ Merged {len(folders)} experiments into:\n{file_names}{overlap_note}"
        )

        # Stay open so the user can run further merges without relaunching, but
        # disable Merge until the selection, output directory, or file name prefix
        # changes (avoids re-merging the exact same set to the same output).
        self._last_merged_state = (
            {str(f) for f in folders},
            self.edit_out.text(),
            self.edit_filename.text(),
        )
        self._update_merge_enabled()


# %% launch_merge_dialog() ----


def launch_merge_dialog() -> None:
    """Open the standalone element-list merge window."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = MergeElemLists()
    dialog.exec_()
