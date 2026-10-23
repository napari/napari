from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from qtpy import QtCore
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from napari.utils.progress import cancelable_progress, progress

if TYPE_CHECKING:
    from napari.utils.events import Event


class QtLabeledProgressBar(QWidget):
    """QProgressBar with QLabels for description and ETA."""

    def __init__(
        self, parent: QWidget | None = None, prog: progress | None = None
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.progress = prog

        self.qt_progress_bar = QProgressBar()
        self.description_label = QLabel()
        self.eta_label = QLabel()
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setVisible(isinstance(prog, cancelable_progress))
        base_layout = QVBoxLayout()

        pbar_layout = QHBoxLayout()
        pbar_layout.addWidget(self.description_label)
        pbar_layout.addWidget(self.qt_progress_bar)
        pbar_layout.addWidget(self.eta_label)
        pbar_layout.addWidget(self.cancel_button)
        base_layout.addLayout(pbar_layout)

        line = QFrame(self)
        line.setObjectName('QtCustomTitleBarLine')
        line.setFixedHeight(1)
        base_layout.addWidget(line)

        self.setLayout(base_layout)

        self._pending_updates: dict[str, Any] = {}
        self._pending_lock = threading.Lock()
        self._applying_updates = False

    def setRange(self, min_val: int, max_val: int) -> None:
        self.qt_progress_bar.setRange(min_val, max_val)

    def setValue(self, value: int) -> None:
        self.qt_progress_bar.setValue(value)
        QApplication.processEvents()

    def setDescription(self, value: str) -> None:
        if not value.endswith(': '):
            value = f'{value}: '
        self.description_label.setText(value)
        QApplication.processEvents()

    def _set_value(self, event: Event) -> None:
        self._queue_update('value', event.value)

    def _get_value(self) -> int:
        return self.qt_progress_bar.value()

    def _set_description(self, event: Event) -> None:
        self._queue_update('description', event.value)

    def _make_indeterminate(self, event: Event) -> None:
        self._queue_update('range', (0, 0))

    def _set_eta(self, event: Event) -> None:
        self._queue_update('eta', event.value)

    def _set_total(self, event: Event) -> None:
        self._queue_update('range', (0, event.value))

    def _queue_update(self, kind: str, value: Any) -> None:
        # Updates may come fast from a background thread. Keep only the
        # latest ones and apply them together on the main thread.
        with self._pending_lock:
            needs_apply = not self._pending_updates
            self._pending_updates[kind] = value
        if needs_apply:
            self._apply_pending_updates()

    @ensure_main_thread
    def _apply_pending_updates(self) -> None:
        # Qt events processed in `setValue` can call this again. Let the
        # running call do the work, so calls never pile up.
        if self._applying_updates:
            return
        self._applying_updates = True
        try:
            while True:
                with self._pending_lock:
                    updates, self._pending_updates = self._pending_updates, {}
                if not updates:
                    break
                self._apply_updates(updates)
        except RuntimeError as e:
            # A last update (e.g. the ETA sent when `progress` closes) can
            # arrive after the bar was closed and deleted. Ignore it.
            if 'C++ object' not in str(e):
                raise
        finally:
            self._applying_updates = False

    def _apply_updates(self, updates: dict[str, Any]) -> None:
        if 'range' in updates:
            self.setRange(*updates['range'])
        if 'value' in updates:
            self.setValue(updates['value'])
        if 'description' in updates:
            self.setDescription(updates['description'])
        if 'eta' in updates:
            self.eta_label.setText(updates['eta'])

    def _cancel(self) -> None:
        if self.progress is None:
            return
        self.cancel_button.setText('Cancelling...')
        self.progress.cancel()
        self.cancel_button.setText('Canceled')


class QtProgressBarGroup(QWidget):
    """One or more QtLabeledProgressBars with a QFrame line separator at the bottom"""

    def __init__(
        self,
        qt_progress_bar: QtLabeledProgressBar,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        pbr_group_layout = QVBoxLayout()
        pbr_group_layout.addWidget(qt_progress_bar)
        pbr_group_layout.setContentsMargins(0, 0, 0, 0)

        line = QFrame(self)
        line.setObjectName('QtCustomTitleBarLine')
        line.setFixedHeight(1)
        pbr_group_layout.addWidget(line)

        self.setLayout(pbr_group_layout)
