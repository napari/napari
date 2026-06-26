import numpy as np
from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Image, Labels
from napari.utils.misc import human_readable_size
from napari.utils.translations import trans


def _format_level_label(
    index: int,
    shape: tuple,
    nbytes: int,
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

    Returns
    -------
    str
        e.g. ``"0: 256 x 256 x 128 (8.4 MB)"``
    """
    shape_str = ' \u00d7 '.join(str(s) for s in shape)
    size_str = human_readable_size(nbytes)
    return f'{index}: {shape_str} ({size_str})'


# MetaWidgetControlsBase merges type(QObject) and type(ABC) at runtime,
# but mypy cannot verify this is safe.
class QtMultiscaleLevelControl(  # type: ignore[metaclass]
    QtWidgetControlsBase,
):
    """Widget to manually select which multiscale level to render.

    Shows a combobox with "Auto" plus one entry per resolution level.
    Only visible when the layer is multiscale.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    layer : Image | Labels
        A napari layer with multiscale data (Image or Labels).

    Attributes
    ----------
    level_combobox : QComboBox
        Combobox listing "Auto" and each resolution level.
    level_label : QtWrappedLabel
        Label for the resolution combobox.
    """

    def __init__(self, parent: QWidget, layer: Image | Labels) -> None:
        super().__init__(parent, layer)
        self._layer: Image | Labels = layer

        self.level_combobox = QComboBox(parent)
        self.level_label = QtWrappedLabel(trans._('resolution:'))

        # Only set up and show widgets if layer is multiscale
        if layer.multiscale:
            self._update_level_labels()
            self.level_combobox.currentIndexChanged.connect(
                self._on_combobox_changed
            )
            self._layer.events.locked_data_level.connect(
                self._on_locked_data_level_change
            )
            self._layer.events.data.connect(self._update_level_labels)
            self.level_combobox.show()
            self.level_label.show()
        else:
            self.level_combobox.hide()
            self.level_label.hide()

    def _update_level_labels(self) -> None:
        """Populate the combobox with resolution level labels."""
        with qt_signals_blocked(self.level_combobox):
            self.level_combobox.clear()
            self.level_combobox.addItem('Auto', None)

            if self._layer.multiscale:
                shapes = self._layer.level_shapes
                itemsize = self._layer.dtype.itemsize
                for i, shape in enumerate(shapes):
                    # Calculate size using full shape
                    nbytes = int(np.prod(shape) * itemsize)

                    label = _format_level_label(i, tuple(shape), nbytes)
                    self.level_combobox.addItem(label, i)

            # Reflect current locked state
            locked = getattr(self._layer, '_locked_data_level', None)
            if locked is not None:
                # +1 because index 0 is "Auto"
                self.level_combobox.setCurrentIndex(locked + 1)
            else:
                self.level_combobox.setCurrentIndex(0)

    def _on_combobox_changed(self, index: int) -> None:
        """Update the layer's locked data level from the combobox selection.

        Parameters
        ----------
        index : int
            Index of the selected combobox item. ``0`` corresponds to
            "Auto" (``None``); higher indices map to resolution levels.
        """
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

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        """Return the label/widget pairs for this control.

        Returns
        -------
        list[tuple[QtWrappedLabel, QWidget]]
            Single-element list containing the resolution label and combobox.
        """
        return [(self.level_label, self.level_combobox)]
