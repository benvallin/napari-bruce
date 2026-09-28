# %% Import required libraries ----

import sys
import traceback
from pathlib import Path

from qtpy.QtCore import Qt, QObject, QThread, Signal
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


# %% MergeWorker() ----


class MergeWorker(QObject):
    """Run workflow.merge_elem_lists() off the UI thread.

    merge_elem_lists() does an O(N^2) pairwise contour-overlap scan over every
    cut element being merged; for many folders/thousands of elements this can
    take seconds, and running it synchronously on a button click would freeze
    the whole napari GUI (this dialog has no other worker infrastructure, so a
    small bespoke QObject/QThread pair is used here rather than the fuller
    generic worker machinery in main_widget.py).
    """

    sig_success = Signal(tuple)
    sig_error = Signal(str)

    def __init__(
        self,
        folder_paths,
        exclude_nos: set[int] | None = None,
        detect_overlaps: bool = True,
        parent=None,
    ):

        super().__init__(parent)
        self.folder_paths = folder_paths
        self.exclude_nos = exclude_nos
        self.detect_overlaps = detect_overlaps

    def run(self):

        try:

            result = workflow.merge_elem_lists(
                folder_paths=self.folder_paths,
                exclude_nos=self.exclude_nos,
                detect_overlaps=self.detect_overlaps,
            )
            self.sig_success.emit(result)

        except Exception as e:

            # Print the full traceback to the terminal for debuggability (matching
            # main_widget.py's BaseWorker.run()); the dialog itself only needs the
            # short message for its QMessageBox.
            traceback.print_exc()
            self.sig_error.emit(f"{type(e).__name__}: {e}")


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

        # True while a background merge computation is in flight — keeps Merge
        # disabled regardless of what _update_merge_enabled would otherwise
        # compute, so a checkbox/field change during the compute can't re-enable
        # it and let the user fire a second, overlapping merge.
        self._merge_in_progress = False
        self._merge_worker_thread: QThread | None = None
        self._merge_worker: "MergeWorker | None" = None

        # Stashed by _on_merge, consumed by _finalize_merge once the chained
        # background pass(es) complete -- see _start_merge_worker.
        self._merge_folders: list[Path] = []
        self._merge_out_dir: Path | None = None
        self._merge_prefix: str = ""
        self._merge_base: str = ""

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

        if self._merge_in_progress:
            self.btn_merge.setEnabled(False)
            return

        selected = {str(f) for f in self._selected_folders()}
        state = (selected, self.edit_out.text(), self.edit_filename.text())
        unchanged = state == self._last_merged_state
        self.btn_merge.setEnabled(len(selected) >= 2 and not unchanged)

    def closeEvent(self, event):

        # Refuse to close while a merge is running, rather than blocking here
        # with thread.wait() (freezes the GUI for the remainder of the
        # O(N^2) overlap scan -- the exact synchronous freeze MergeWorker was
        # introduced to avoid) or letting the dialog get destroyed while a
        # queued worker signal could still be delivered to it afterward (a
        # "wrapped C/C++ object has been deleted" crash risk).
        if isinstance(self._merge_worker_thread, QThread) and self._merge_worker_thread.isRunning():
            self.status_label.setText(
                "A merge is currently in progress. Please wait for it to "
                "finish before closing this window."
            )
            event.ignore()
            return

        super().closeEvent(event)

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

    def _start_merge_worker(
        self, folder_paths, on_done, exclude_nos=None, detect_overlaps=True
    ) -> None:
        """Run workflow.merge_elem_lists() on a background QThread and route its
        result to 'on_done' (or _on_merge_error on failure)."""

        thread = QThread()
        worker = MergeWorker(
            folder_paths, exclude_nos=exclude_nos, detect_overlaps=detect_overlaps
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.sig_success.connect(on_done)
        worker.sig_error.connect(self._on_merge_error)
        worker.sig_success.connect(thread.quit)
        worker.sig_error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        def reset_attrs():
            # Only clear if these attrs still refer to THIS thread/worker: on the
            # two-pass merge, pass 1's on_done can synchronously start pass 2
            # (reassigning self._merge_worker_thread/_merge_worker) before pass
            # 1's own 'finished' (queued from the same emission, delivered after
            # on_done) fires this closure -- without the identity check, pass
            # 1's finished would null out pass 2's still-live references.
            if self._merge_worker_thread is thread:
                self._merge_worker_thread = None
            if self._merge_worker is worker:
                self._merge_worker = None

        thread.finished.connect(reset_attrs)

        self._merge_worker_thread = thread
        self._merge_worker = worker
        thread.start()

    def _end_merge_in_progress(self) -> None:
        self._merge_in_progress = False
        self._update_merge_enabled()

    def _on_merge_error(self, error_msg: str) -> None:
        self._end_merge_in_progress()
        QMessageBox.critical(self, "Merge failed", error_msg)

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

        # Stashed for _finalize_merge, called once the background pass(es) below
        # complete -- merge_elem_lists() does an O(N^2) overlap scan, so it runs
        # off the UI thread (see MergeWorker), possibly twice if the user excludes
        # overlapping elements via the modal resolution dialog in between.
        self._merge_folders = folders
        self._merge_out_dir = out_dir
        self._merge_prefix = prefix
        self._merge_base = base

        self._merge_in_progress = True
        self.btn_merge.setEnabled(False)
        self.status_label.setText("Merging…")

        self._start_merge_worker(folders, on_done=self._on_first_merge_pass_done)

    def _on_first_merge_pass_done(self, result: tuple) -> None:

        merged, overlaps, collected, omitted = result

        # Cut elements from different images whose contours overlap signal
        # partially overlapping FOVs → the same tissue could be captured twice.
        # Show the overlap resolution dialog so the user can inspect each pair
        # and choose which elements to keep before writing.
        if overlaps:
            warning = workflow.format_overlap_warnings(overlaps)
            _safe_print(warning + "\n")
            resolution = OverlapResolutionDialog(overlaps, parent=self)
            if resolution.exec_() != QDialog.Accepted:
                self._end_merge_in_progress()
                self.status_label.setText(
                    f"✘ Merge cancelled: {len(overlaps)} overlapping element "
                    "pair(s) across images."
                )
                return

            excluded = resolution.excluded_nos()

            if excluded:
                # Re-run with the exclusions applied; overlaps were already found
                # above, so skip re-scanning for them on this pass.
                self._start_merge_worker(
                    self._merge_folders,
                    on_done=lambda result, overlaps=overlaps, excluded=excluded: (
                        self._on_second_merge_pass_done(result, overlaps, excluded)
                    ),
                    exclude_nos=excluded,
                    detect_overlaps=False,
                )
                return

            self._finalize_merge(merged, collected, omitted, overlaps, excluded)
            return

        self._finalize_merge(merged, collected, omitted, overlaps, set())

    def _on_second_merge_pass_done(
        self, result: tuple, overlaps: list, excluded: set
    ) -> None:

        merged, _, collected, omitted = result
        self._finalize_merge(merged, collected, omitted, overlaps, excluded)

    def _finalize_merge(
        self, merged: dict, collected: list, omitted: list, overlaps: list, excluded: set
    ) -> None:

        folders = self._merge_folders
        out_dir = self._merge_out_dir
        prefix = self._merge_prefix
        base = self._merge_base

        # Wrapped in try/except (unlike the pre-threading synchronous version,
        # an uncaught exception here would leave self._merge_in_progress stuck
        # True -- see _end_merge_in_progress -- permanently disabling Merge for
        # the rest of the dialog's life, worse than a normal re-enable would
        # have been after a failure).
        try:

            # Always suffix the group name (collect / omit / all), mirroring
            # single-run output (elem_list_collect.txt, ...), so a collect-only
            # merge still gets the "_collect" suffix. To avoid overwriting a
            # previous merge, bump a numeric tag "(1)", "(2)", ... applied to the
            # whole group so the set stays together (collect/omit/all/
            # collection_summary share one index) and no member collides with an
            # existing file.
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

        except Exception as e:
            self._on_merge_error(f"{type(e).__name__}: {e}")
            return

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
        self._end_merge_in_progress()


# %% launch_merge_dialog() ----


def launch_merge_dialog() -> None:
    """Open the standalone element-list merge window."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = MergeElemLists()
    dialog.exec_()
