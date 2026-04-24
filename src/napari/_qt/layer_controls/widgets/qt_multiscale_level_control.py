import numpy as np
from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.base.base import Layer
from napari.utils.translations import trans


def _human_readable_size(size_bytes: float) -> str:
    """Convert bytes to human-readable string (KB, MB, GB, etc.)."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(size_bytes) < 1000:
            return f'{size_bytes:.1f} {unit}'
        size_bytes /= 1000
    return f'{size_bytes:.1f} PB'


def _format_level_label(
    index: int,
    shape: tuple,
    nbytes: int,
    displayed_axes: tuple[int, ...] | None = None,
) -> str:
    """Build a human-readable label for one multiscale level.

    Parameters
    ----------
    index : int
        Zero-based level index.
    shape : tuple of int
        Full shape of the array at this level.
    nbytes : int
        Size of the array in bytes.
    displayed_axes : tuple of int, optional
        Indices of the currently displayed dimensions.  When provided only
        those dimensions are shown in the label; otherwise the full shape
        is used.

    Returns
    -------
    str
        e.g. ``"0: 256 × 256 × 128 (8.4 MB)"``
    """
    if displayed_axes is not None:
        dims = tuple(shape[ax] for ax in displayed_axes)
    else:
        dims = shape
    shape_str = ' \u00d7 '.join(str(s) for s in dims)
    size_str = _human_readable_size(nbytes)
    return f'{index}: {shape_str} ({size_str})'


class QtMultiscaleLevelControl(QtWidgetControlsBase):
    """Widget to manually select which multiscale level to render.

    Shows a combobox with "Auto" plus one entry per resolution level.
    Only visible when the layer is multiscale.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    layer : Layer
        A napari layer with multiscale data (Image or Labels).
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)

        self.level_combobox = QComboBox(parent)
        self.level_label = QtWrappedLabel(trans._('resolution:'))

        self._rebuild_items()
        self.level_combobox.currentIndexChanged.connect(
            self._on_combobox_changed
        )
        self._layer.events.locked_data_level.connect(
            self._on_locked_data_level_change
        )

    def _rebuild_items(self) -> None:
        """Populate the combobox from the layer's current level_shapes."""
        displayed = tuple(self._layer._slice_input.displayed)
        with qt_signals_blocked(self.level_combobox):
            self.level_combobox.clear()
            self.level_combobox.addItem('Auto', None)

            if self._layer.multiscale:
                shapes = self._layer.level_shapes
                for i, shape in enumerate(shapes):
                    data_arr = self._layer.data[i]
                    if hasattr(data_arr, 'nbytes'):
                        nbytes = data_arr.nbytes
                    else:
                        dtype = getattr(data_arr, 'dtype', np.float32)
                        nbytes = int(
                            np.prod(shape) * np.dtype(dtype).itemsize
                        )

                    label = _format_level_label(
                        i, tuple(shape), nbytes, displayed
                    )
                    self.level_combobox.addItem(label, i)

            # Reflect current locked state
            locked = getattr(self._layer, '_locked_data_level', None)
            if locked is not None:
                # +1 because index 0 is "Auto"
                self.level_combobox.setCurrentIndex(locked + 1)
            else:
                self.level_combobox.setCurrentIndex(0)

    def _on_combobox_changed(self, index: int) -> None:
        level = self.level_combobox.itemData(index)
        self._layer.locked_data_level = level

    def _on_locked_data_level_change(self) -> None:
        """Sync the combobox when locked_data_level is set programmatically."""
        locked = self._layer.locked_data_level
        with qt_signals_blocked(self.level_combobox):
            if locked is not None:
                self.level_combobox.setCurrentIndex(locked + 1)
            else:
                self.level_combobox.setCurrentIndex(0)

    def _on_display_change_show(self) -> None:
        if self._layer.multiscale:
            self._rebuild_items()
            self.level_combobox.show()
            self.level_label.show()

    def _on_display_change_hide(self) -> None:
        self.level_combobox.hide()
        self.level_label.hide()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.level_label, self.level_combobox)]
