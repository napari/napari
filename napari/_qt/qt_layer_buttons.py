from qtpy.QtWidgets import QHBoxLayout, QPushButton, QFrame


class QtLayersButtons(QFrame):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.deleteButton = QtDeleteButton(self.viewer)
        self.newPointsButton = QtNewPointsButton(self.viewer)
        self.newShapesButton = QtNewShapesButton(self.viewer)
        self.newLabelsButton = QtNewLabelsButton(self.viewer)

        layout = QHBoxLayout()
        layout.addStretch(0)
        layout.setContentsMargins(0, 18, 34, 4)
        layout.addWidget(self.newPointsButton)
        layout.addWidget(self.newShapesButton)
        layout.addWidget(self.newLabelsButton)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)


class QtDeleteButton(QPushButton):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete layers')
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


class QtNewPointsButton(QPushButton):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('New points layer')
        self.clicked.connect(lambda: self.viewer._new_points())


class QtNewShapesButton(QPushButton):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('New shapes layer')
        self.clicked.connect(lambda: self.viewer._new_shapes())


class QtNewLabelsButton(QPushButton):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('New labels layer')
        self.clicked.connect(lambda: self.viewer._new_labels())
