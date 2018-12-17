from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QCheckBox, QScrollArea
from os.path import dirname, join, realpath
from numpy import empty


dir_path = dirname(realpath(__file__))
path_delete = join(dir_path,'icons','delete.png')
path_add = join(dir_path,'icons','add.png')
path_off = join(dir_path,'icons','annotation_off.png')
path_on = join(dir_path,'icons','annotation_on.png')

class QtLayerPanel(QWidget):
    def __init__(self, layers):
        super().__init__()

        layout = QVBoxLayout()
        self.layersList = QtLayerList(layers)
        self.layersControls = QtLayerButtons(layers)
        layout.addWidget(self.layersList)
        layout.addWidget(self.layersControls)
        self.setLayout(layout)

class QtLayerButtons(QFrame):
    def __init__(self, layers):
        super().__init__()

        self.annotationCheckBox = QAnnotationCheckBox(layers)
        layout = QHBoxLayout()
        layout.addWidget(self.annotationCheckBox)
        layout.addStretch(0)
        layout.addWidget(QAddLayerButton(layers))
        layout.addWidget(QDeleteButton(layers))

        self.setLayout(layout)

class QtLayerList(QScrollArea):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setWidgetResizable(True)
        #self.setFixedWidth(315)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollWidget = QWidget()
        self.setWidget(scrollWidget)
        self.layersLayout = QVBoxLayout(scrollWidget)
        self.layersLayout.addWidget(QtDivider())
        self.layersLayout.addStretch(1)
        self.setAcceptDrops(True)
        self.setToolTip('Layer list')

    def insert(self, index, total, layer):
        """Inserts a layer widget at a specific index
        """
        if layer._qt is not None:
            self.layersLayout.insertWidget(2*(total - index)-1, layer._qt)
            self.layersLayout.insertWidget(2*(total - index), QtDivider())
        self.layers.viewer._update_active_layers()
        self.layers.viewer.controlBars.climSliderUpdate()

    def remove(self, layer):
        """Removes a layer widget
        """
        if layer._qt is not None:
            index = self.layersLayout.indexOf(layer._qt)
            divider = self.layersLayout.itemAt(index+1).widget()
            self.layersLayout.removeWidget(layer._qt)
            layer._qt.deleteLater()
            layer._qt = None
            self.layersLayout.removeWidget(divider)
            divider.deleteLater()
            divider = None
        self.layers.viewer._update_active_layers()
        self.layers.viewer.controlBars.climSliderUpdate()

    def reorder(self):
        """Reorders list of layer widgets by looping through all
        widgets in list sequentially removing them and inserting
        them into the correct place in final list.
        """
        total = len(self.layers)
        for i in range(total):
            layer = self.layers[i]
            if layer._qt is not None:
                index = self.layersLayout.indexOf(layer._qt)
                divider = self.layersLayout.itemAt(index+1).widget()
                self.layersLayout.removeWidget(layer._qt)
                self.layersLayout.removeWidget(divider)
                self.layersLayout.insertWidget(2*(total - i)-1,layer._qt)
                self.layersLayout.insertWidget(2*(total - i),divider)
        self.layers.viewer._update_active_layers()
        self.layers.viewer.controlBars.climSliderUpdate()

    def mouseReleaseEvent(self, event):
        """Unselects all layer widgets
        """
        if self.layersLayout.count() > 1:
            self.layersLayout.itemAt(1).widget().unselectAll()
        self.layers.viewer._update_active_layers()
        self.layers.viewer._set_annotation_mode(self.layers.viewer.annotation)
        self.layers.viewer.controlBars.climSliderUpdate()
        self.layers.viewer._status = 'Ready'
        self.layers.viewer.emitStatus()

    def dragLeaveEvent(self, event):
        event.ignore()
        for i in range(0, self.layersLayout.count(), 2):
            self.layersLayout.itemAt(i).widget().setSelected(False)

    def dragEnterEvent(self, event):
        event.accept()
        dividers = []
        for i in range(0, self.layersLayout.count(), 2):
            widget = self.layersLayout.itemAt(i).widget()
            dividers.append(widget.y()+widget.frameGeometry().height()/2)
        self.centers = [(dividers[i+1]+dividers[i])/2 for i in range(len(dividers)-1)]

    def dragMoveEvent(self, event):
        cord = event.pos().y()
        divider_index = next((i for i, x in enumerate(self.centers) if x > cord), len(self.centers))
        layerWidget = event.source()
        layers = layerWidget.layer.viewer.layers
        index = layers.index(layerWidget.layer)
        total = len(layers)
        insert_index = total - divider_index
        if not (insert_index == index) and not (insert_index-1 == index):
            state = True
        else:
            state = False
        for i in range(0, self.layersLayout.count(), 2):
            if i == 2*divider_index:
                self.layersLayout.itemAt(i).widget().setSelected(state)
            else:
                self.layersLayout.itemAt(i).widget().setSelected(False)

    def dropEvent(self, event):
        for i in range(0, self.layersLayout.count(), 2):
            self.layersLayout.itemAt(i).widget().setSelected(False)
        cord = event.pos().y()
        divider_index = next((i for i, x in enumerate(self.centers) if x > cord), len(self.centers))
        layerWidget = event.source()
        layers = layerWidget.layer.viewer.layers
        index = layers.index(layerWidget.layer)
        total = len(layers)
        insert_index = total - divider_index
        indices = [i for i in range(total)]
        if layerWidget.layer.selected:
            selected = []
            for i in range(total):
                if layers[i].selected:
                    selected.append(i)
        else:
            selected = [index]
        for i in selected:
            indices.remove(i)
        offset = sum([i<insert_index for i in selected])
        j = insert_index - offset
        for i in selected:
            indices.insert(j,i)
            j = j+1
        if not indices == [i for i in range(total)]:
            layers.reorder(indices)
            event.accept()
        else:
            event.ignore()
        if not layerWidget.layer.selected:
            layerWidget.unselectAll()
            layerWidget.setSelected(True)

class QtDivider(QFrame):
    def __init__(self):
        super().__init__()
        self.unselectedStlyeSheet = "QFrame {border: 3px solid rgb(236,236,236); background-color:rgb(236,236,236); border-radius: 3px;}"
        self.selectedStlyeSheet = "QFrame {border: 3px solid rgb(0, 153, 255); background-color:rgb(0, 153, 255); border-radius: 3px;}"
        self.setSelected(False)
        self.setFixedHeight(4)

    def setSelected(self, bool):
        if bool:
            self.setStyleSheet(self.selectedStlyeSheet)
        else:
            self.setStyleSheet(self.unselectedStlyeSheet)

class QDeleteButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(path_delete))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete layers')
        self.clicked.connect(self.on_click)
        self.setAcceptDrops(True)
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)

    def on_click(self):
        self.layers.remove_selected()

    def dragEnterEvent(self, event):
        event.accept()
        self.hover = True
        self.update()

    def dragLeaveEvent(self, event):
        event.ignore()
        self.hover = False
        self.update()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        event.accept()

class QAddLayerButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(path_add))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Add layer')
        self.clicked.connect(self.on_click)
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)

    def on_click(self):
        if self.layers.viewer.dimensions.max_dims == 0:
            empty_markers = empty((0, 2))
        else:
            empty_markers = empty((0, self.layers.viewer.dimensions.max_dims))
        self.layers.viewer.add_markers(empty_markers)

class QAnnotationCheckBox(QCheckBox):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setToolTip('Annotation mode')
        self.setChecked(False)
        self.stateChanged.connect(lambda state=self: self.changeAnnotation(state))
        styleSheet = """QCheckBox {background-color:lightGray; border-radius: 3px;}
                        QCheckBox::indicator {subcontrol-position: center center; subcontrol-origin: content;
                            width: 28px; height: 28px;}
                        QCheckBox::indicator:checked {background-color:rgb(0, 153, 255); border-radius: 3px;
                            image: url(""" + path_off + """);}
                        QCheckBox::indicator:unchecked {image: url(""" + path_off + """);}
                        QCheckBox::indicator:unchecked:hover {image: url(""" + path_on + ");}"
        self.setStyleSheet(styleSheet)

    def changeAnnotation(self, state):
        if state == Qt.Checked:
            self.layers.viewer._set_annotation_mode(True)
        else:
            self.layers.viewer._set_annotation_mode(False)
