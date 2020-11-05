"""RenderSpinBox class.
"""
from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QWidget


class QtLabeledSpinBox(QWidget):
    """A label plus a SpinBox for the QtRender widget.

    This was cobbled together quickly for QtRender. We could probably use
    some napari-standard control instead?

    Parameters
    ----------
    label_text : str
        The label shown next to the spin box.
    initial_value : int
        The initial value of the spin box.
    spin_range : range
        The min/max/step of the spin box.
    connect : Callable[[int], None]
        Called when the user changes the value of the spin box.

    Attributes
    ----------
    spin : QSpinBox
        The spin box.
    """

    def __init__(
        self,
        label_text: str,
        initial_value: int,
        spin_range: range,
        connect: Callable[[int], None] = None,
    ):
        super().__init__()
        self.connect = connect
        layout = QHBoxLayout()

        self.spin = self._create(initial_value, spin_range)

        layout.addWidget(QLabel(label_text))
        layout.addWidget(self.spin)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _create(self, initial_value: int, spin_range: range) -> QSpinBox:
        """Return the configured QSpinBox.

        Parameters
        ----------
        initial_value : int
            The initial value of the QSpinBox.
        spin_range : range
            The start/stop/step of the QSpinBox.

        Return
        ------
        QSpinBox
            The configured QSpinBox.
        """
        spin = QSpinBox()

        spin.setMinimum(spin_range.start)
        spin.setMaximum(spin_range.stop)
        spin.setSingleStep(spin_range.step)
        spin.setValue(initial_value)

        spin.setKeyboardTracking(False)  # correct?
        spin.setAlignment(Qt.AlignCenter)

        spin.valueChanged.connect(self._on_change)
        return spin

    def _on_change(self, value: int) -> None:
        """Called when the spin spin value was changed.

        value : int
            The new value of the SpinBox.
        """
        # We must clearFocus or it would double-step, no idea why.
        self.spin.clearFocus()

        # Notify any connection we have.
        if self.connect is not None:
            self.connect(value)
