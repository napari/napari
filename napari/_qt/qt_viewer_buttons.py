from qtpy.QtWidgets import QHBoxLayout, QPushButton, QFrame, QCheckBox
from qtpy.QtCore import Qt
import numpy as np


class QtLayerButtons(QFrame):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.deleteButton = QtDeleteButton(self.viewer)
        self.newPointsButton = QtViewerPushButton(
            self.viewer,
            'New points layer',
            'new_points',
            lambda: self.viewer.add_points(data=None),
        )
        self.newShapesButton = QtViewerPushButton(
            self.viewer,
            'New shapes layer',
            'new_shapes',
            lambda: self.viewer.add_shapes(data=None),
        )
        self.newLabelsButton = QtViewerPushButton(
            self.viewer,
            'New labels layer',
            'new_labels',
            lambda: self.viewer._new_labels(),
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.newPointsButton)
        layout.addWidget(self.newShapesButton)
        layout.addWidget(self.newLabelsButton)
        layout.addStretch(0)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)


class QtViewerButtons(QFrame):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.consoleButton = QtViewerPushButton(
            self.viewer, 'Open IPython terminal', 'console'
        )
        self.consoleButton.setProperty('expanded', False)
        self.rollDimsButton = QtViewerPushButton(
            self.viewer,
            'Roll dimensions order for display',
            'roll',
            lambda: self.viewer.dims._roll(),
        )
        self.transposeDimsButton = QtViewerPushButton(
            self.viewer,
            'Transpose displayed dimensions',
            'transpose',
            lambda: self.viewer.dims._transpose(),
        )
        self.resetViewButton = QtViewerPushButton(
            self.viewer, 'Reset view', 'home', lambda: self.viewer.reset_view()
        )
        self.gridViewButton = QtGridViewButton(self.viewer)
        self.ndisplayButton = QtNDisplayButton(self.viewer)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.consoleButton)
        layout.addWidget(self.ndisplayButton)
        layout.addWidget(self.rollDimsButton)
        layout.addWidget(self.transposeDimsButton)
        layout.addWidget(self.gridViewButton)
        layout.addWidget(self.resetViewButton)
        layout.addStretch(0)
        self.setLayout(layout)


class QtDeleteButton(QPushButton):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setToolTip('Delete selected layers')
        self.setAcceptDrops(True)
        self.clicked.connect(lambda: self.viewer.layers.remove_selected())

    def dragEnterEvent(self, event):
        event.accept()
        self.hover = True
        self.update()

    def dragLeaveEvent(self, event):
        event.ignore()
        self.hover = False
        self.update()

    def dropEvent(self, event):
        event.accept()
        layer_name = event.mimeData().text()
        layer = self.viewer.layers[layer_name]
        if not layer.selected:
            self.viewer.layers.remove(layer)
        else:
            self.viewer.layers.remove_selected()


class QtViewerPushButton(QPushButton):
    def __init__(self, viewer, tooltip, button_name, slot=None):
        super().__init__()

        self.viewer = viewer
        self.setToolTip(tooltip)
        self.setProperty('mode', button_name)
        if slot is not None:
            self.clicked.connect(slot)


class QtGridViewButton(QCheckBox):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setToolTip('Toggle grid view view')
        self.viewer.events.grid.connect(self._on_grid_change)
        self.stateChanged.connect(self.change_grid)
        self._on_grid_change(None)

    def change_grid(self, state):
        if state == Qt.Checked:
            self.viewer.stack_view()
        else:
            self.viewer.grid_view()

    def _on_grid_change(self, event):
        with self.viewer.events.grid.blocker():
            self.setChecked(np.all(self.viewer.grid_size == (1, 1)))


class QtNDisplayButton(QCheckBox):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setToolTip('Toggle number of displayed dimensions')
        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)

        self.setChecked(self.viewer.dims.ndisplay == 3)
        self.stateChanged.connect(
            lambda state=self: self.change_ndisplay(state)
        )

    def change_ndisplay(self, state):
        if state == Qt.Checked:
            self.viewer.dims.ndisplay = 3
        else:
            self.viewer.dims.ndisplay = 2

    def _on_ndisplay_change(self, event):
        with self.viewer.dims.events.ndisplay.blocker():
            self.setChecked(self.viewer.dims.ndisplay == 3)
