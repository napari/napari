from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QHBoxLayout, QPushButton, QWidget
from superqt import QLabeledSlider

from napari._qt.dialogs.qt_modal import QtPopup


class QMirroredSlidersPopup(QtPopup):
    """A popup window that contains two mirrored labeled sliders and buttons.

    Parameters
    ----------
    parent : QWidget, optional
        Will like be an instance of QtLayerControls.  Note, providing
        parent can be useful to inherit stylesheets.

    Attributes
    ----------
    slider : QLabeledRangeSlider
        Slider widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.toggle_lock = QPushButton()
        self.toggle_lock.setCheckable(True)
        self.toggle_lock.setChecked(True)
        self.toggle_lock.setObjectName('lockButton')

        self.left_slider = QLabeledSlider(Qt.Orientation.Horizontal, parent)
        self.left_slider.setObjectName('invertedSlider')
        self.left_slider.setInvertedAppearance(True)
        self.left_slider.setEdgeLabelPosition(
            QLabeledSlider.LabelPosition.LabelsLeft
        )
        self.right_slider = QLabeledSlider(Qt.Orientation.Horizontal, parent)

        self.left_slider.setFocus()

        # add widgets to layout
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(2, 2, 2, 2)
        self.frame.setLayout(self._layout)
        self._layout.addWidget(self.left_slider)
        self._layout.addWidget(self.toggle_lock)
        self._layout.addWidget(self.right_slider)
        QApplication.processEvents()

        self.left_slider.valueChanged.connect(self._on_left_slider_change)
        self.right_slider.valueChanged.connect(self._on_right_slider_change)

    def _on_left_slider_change(self, value: int) -> None:
        if self.toggle_lock.isChecked() and self.right_slider.value() != value:
            self.right_slider.setValue(value)

    def _on_right_slider_change(self, value: int) -> None:
        if self.toggle_lock.isChecked() and self.left_slider.value() != value:
            self.left_slider.setValue(value)

    def _on_toggle_change(self) -> None:
        if self.toggle_lock.isChecked():
            mean = (self.left_slider.value() + self.right_slider.value()) // 2
            self.left_slider.setValue(mean)
            self.right_slider.setValue(mean)
