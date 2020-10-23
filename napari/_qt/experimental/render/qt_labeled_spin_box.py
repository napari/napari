"""RenderSpinBox class.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSpinBox


class LabeledSpinBox:
    """A label plus a SpinBox for the QtRender widget.

    This was cobbled together quickly for QtRender. We could probably use
    some napari-standard control instead? This is probably not good Qt.
    """

    def __init__(
        self,
        parent,
        label_text: str,
        initial_value: int,
        spin_range: range,
        connect=None,
    ):
        self.connect = connect

        layout = QHBoxLayout()

        layout.addWidget(QLabel(label_text))
        self.spin = self._create(initial_value, spin_range)
        layout.addWidget(self.spin)

        parent.addLayout(layout)

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

    def value(self) -> int:
        """Return the current value of the QSpinBox.

        Return
        ------
        int
            The current value of the SpinBox.
        """
        return self.spin.value()

    def set(self, value: int) -> None:
        """Set the current value of the SpinBox.

        Parameters
        ----------
        value : int
            Set SpinBox to this value.
        """
        self.spin.setValue(value)
