from qtpy.QtWidgets import (
    QSlider,
    QLineEdit,
    QGridLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QPushButton,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap

from .._constants import Blending


class QtLayerProperties(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        layer.events.select.connect(self._on_select)
        layer.events.deselect.connect(self._on_deselect)
        layer.events.name.connect(self._on_layer_name_change)
        layer.events.blending.connect(self._on_blending_change)
        layer.events.opacity.connect(self._on_opacity_change)
        layer.events.visible.connect(self._on_visible_change)
        layer.events.thumbnail.connect(self._on_thumbnail_change)

        self.setObjectName('layer')

        self.vbox_layout = QVBoxLayout()
        self.top = QFrame()
        self.top_layout = QHBoxLayout()
        self.grid = QFrame()
        self.grid_layout = QGridLayout()
        self.vbox_layout.addWidget(self.top)
        self.vbox_layout.addWidget(self.grid)
        self.vbox_layout.setSpacing(0)
        self.top.setFixedHeight(38)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setAlignment(Qt.AlignCenter)
        self.top.setLayout(self.top_layout)
        self.grid.setLayout(self.grid_layout)
        self.setLayout(self.vbox_layout)

        self.name_column = 0
        self.property_column = 1

        cb = QCheckBox(self)
        cb.setObjectName('visibility')
        cb.setToolTip('Layer visibility')
        cb.setChecked(self.layer.visible)
        cb.setProperty('mode', 'visibility')
        cb.stateChanged.connect(lambda state=cb: self.changeVisible(state))
        self.visibleCheckBox = cb
        self.top_layout.addWidget(cb)

        tb = QLabel(self)
        tb.setObjectName('thumbmnail')
        tb.setToolTip('Layer thumbmnail')
        self.thumbnail_label = tb
        self._on_thumbnail_change(None)
        self.top_layout.addWidget(tb)

        textbox = QLineEdit(self)
        textbox.setText(layer.name)
        textbox.home(False)
        textbox.setToolTip('Layer name')
        textbox.setAcceptDrops(False)
        textbox.setEnabled(True)
        textbox.editingFinished.connect(self.changeText)
        self.nameTextBox = textbox
        self.top_layout.addWidget(textbox)

        pb = QPushButton(self)
        pb.setToolTip('Expand properties')
        pb.clicked.connect(self.changeExpanded)
        pb.setObjectName('expand')
        self.expand_button = pb
        self.top_layout.addWidget(pb)

        row = self.grid_layout.rowCount()
        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(self.layer.opacity * 100)
        sld.valueChanged[int].connect(
            lambda value=sld: self.changeOpacity(value)
        )
        self.opacitySilder = sld
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('opacity:'), row, self.name_column)
        self.grid_layout.addWidget(sld, row, self.property_column)

        row = self.grid_layout.rowCount()
        blend_comboBox = QComboBox()
        for blend in Blending:
            blend_comboBox.addItem(str(blend))
        index = blend_comboBox.findText(
            self.layer.blending, Qt.MatchFixedString
        )
        blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(
            lambda text=blend_comboBox: self.changeBlending(text)
        )
        self.blendComboBox = blend_comboBox
        self.grid_layout.addWidget(QLabel('blending:'), row, self.name_column)
        self.grid_layout.addWidget(blend_comboBox, row, self.property_column)

        msg = 'Click to select\nDrag to rearrange\nDouble click to expand'
        self.setToolTip(msg)
        self.setExpanded(False)
        self.setFixedWidth(250)
        self.grid_layout.setColumnMinimumWidth(0, 100)
        self.grid_layout.setColumnMinimumWidth(1, 100)

    def _on_select(self, event):
        self.setProperty('selected', True)
        self.nameTextBox.setEnabled(True)
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_deselect(self, event):
        self.setProperty('selected', False)
        self.nameTextBox.setEnabled(False)
        self.style().unpolish(self)
        self.style().polish(self)

    def changeOpacity(self, value):
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value / 100

    def changeVisible(self, state):
        if state == Qt.Checked:
            self.layer.visible = True
        else:
            self.layer.visible = False

    def changeText(self):
        self.layer.name = self.nameTextBox.text()

    def changeBlending(self, text):
        self.layer.blending = text

    def mouseReleaseEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        self.setExpanded(not self.expanded)

    def changeExpanded(self):
        self.setExpanded(not self.expanded)

    def setExpanded(self, bool):
        if bool:
            self.expanded = True
            self.expand_button.setProperty('expanded', True)
            rows = self.grid_layout.rowCount()
            self.setFixedHeight(38 + 30 * rows)
            self.grid.show()
        else:
            self.expanded = False
            self.expand_button.setProperty('expanded', False)
            self.setFixedHeight(60)
            self.grid.hide()
        self.expand_button.style().unpolish(self.expand_button)
        self.expand_button.style().polish(self.expand_button)

    def _on_layer_name_change(self, event):
        with self.layer.events.name.blocker():
            self.nameTextBox.setText(self.layer.name)
            self.nameTextBox.home(False)

    def _on_opacity_change(self, event):
        with self.layer.events.opacity.blocker():
            self.opacitySilder.setValue(self.layer.opacity * 100)

    def _on_blending_change(self, event):
        with self.layer.events.blending.blocker():
            index = self.blendComboBox.findText(
                self.layer.blending, Qt.MatchFixedString
            )
            self.blendComboBox.setCurrentIndex(index)

    def _on_visible_change(self, event):
        with self.layer.events.visible.blocker():
            self.visibleCheckBox.setChecked(self.layer.visible)

    def _on_thumbnail_change(self, event):
        thumbnail = self.layer.thumbnail
        # Note that QImage expects the image width followed by height
        image = QImage(
            thumbnail,
            thumbnail.shape[1],
            thumbnail.shape[0],
            QImage.Format_RGBA8888,
        )
        self.thumbnail_label.setPixmap(QPixmap.fromImage(image))
