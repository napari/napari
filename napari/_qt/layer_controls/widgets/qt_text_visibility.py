from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QCheckBox, QLabel, QWidget

from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtTextVisibilityControl(QObject):
    def __init__(self, parent: QWidget, layer: Layer):
        super().__init__(parent)
        # Setup layer
        self.layer = layer
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        # Setup widgets
        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('toggle text visibility'))
        text_disp_cb.setChecked(self.layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb
        self.textDispLabel = QLabel(trans._('display text:'))

    def change_text_visibility(self, state):
        """Toggle the visibility of the text.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of textDispCheckBox
        """
        self.layer.text.visible = Qt.CheckState(state) == Qt.CheckState.Checked

    def _on_text_visibility_change(self):
        """Receive layer model text visibiltiy change change event and update checkbox."""
        with self.layer.text.events.visible.blocker():
            self.textDispCheckBox.setChecked(self.layer.text.visible)

    def get_widget_controls(self):
        """
        Enable access to the created labels and control widgets.

        Returns
        -------
        list
            List of tuples of the label and widget controls available.

        """
        return [(self.textDispLabel, self.textDispCheckBox)]
