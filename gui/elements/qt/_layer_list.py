from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QCheckBox, QScrollArea

class QtLayerList(QScrollArea):

    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setWidgetResizable(True)
        #self.setFixedWidth(315)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollWidget = QWidget()
        self.setWidget(scrollWidget)
        self.vbox_layout = QVBoxLayout(scrollWidget)
        self.vbox_layout.addWidget(QtDivider())
        self.vbox_layout.addStretch(1)
        self.centers = []
        self.setAcceptDrops(True)
        self.setToolTip('Layer list')

    def insert(self, index, total, layer):
        """Inserts a layer widget at a specific index
        """
        if layer._qt is not None:
            self.vbox_layout.insertWidget(2*(total - index)-1, layer._qt)
            self.vbox_layout.insertWidget(2*(total - index), QtDivider())
            self.layers.viewer._update_active_layers()


    def remove(self, layer):
        """Removes a layer widget
        """
        if layer._qt is not None:
            index = self.vbox_layout.indexOf(layer._qt)
            divider = self.vbox_layout.itemAt(index+1).widget()
            self.vbox_layout.removeWidget(layer._qt)
            layer._qt.deleteLater()
            layer._qt = None
            self.vbox_layout.removeWidget(divider)
            divider.deleteLater()
            divider = None
            self.layers.viewer._update_active_layers()

    def reorder(self, layers):
        """Reorders list of layer widgets by looping through all
        widgets in list sequentially removing them and inserting
        them into the correct place in final list.
        """
        total = len(layers)
        for i in range(total):
            layer = layers[i]
            if layer._qt is not None:
                index = self.vbox_layout.indexOf(layer._qt)
                divider = self.vbox_layout.itemAt(index+1).widget()
                self.vbox_layout.removeWidget(layer._qt)
                self.vbox_layout.removeWidget(divider)
                self.vbox_layout.insertWidget(2*(total - i)-1,layer._qt)
                self.vbox_layout.insertWidget(2*(total - i),divider)
        self.layers.viewer._update_active_layers()

    def mouseReleaseEvent(self, event):
        """Unselects all layer widgets
        """
        self.layers.viewer._update_active_layers()
        self.layers._unselect_all()
        self.layers.viewer._reset()

    def dragLeaveEvent(self, event):
        """Unselects layer dividers
        """
        event.ignore()
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dragEnterEvent(self, event):
        event.accept()
        dividers = []
        for i in range(0, self.vbox_layout.count(), 2):
            widget = self.vbox_layout.itemAt(i).widget()
            dividers.append(widget.y()+widget.frameGeometry().height()/2)
        self.centers = [(dividers[i+1]+dividers[i])/2 for i in range(len(dividers)-1)]

    def dragMoveEvent(self, event):
        cord = event.pos().y()
        divider_index = next((i for i, x in enumerate(self.centers) if x > cord), len(self.centers))
        layerWidget = event.source()
        total = self.vbox_layout.count()//2 - 1
        index = total - self.vbox_layout.indexOf(layerWidget)//2 - 1
        insert = total - divider_index
        if not (insert == index) and not (insert-1 == index):
            state = True
        else:
            state = False
        for i in range(0, self.vbox_layout.count(), 2):
            if i == 2*divider_index:
                self.vbox_layout.itemAt(i).widget().setSelected(state)
            else:
                self.vbox_layout.itemAt(i).widget().setSelected(False)

    def dropEvent(self, event):
        for i in range(0, self.vbox_layout.count(), 2):
            self.vbox_layout.itemAt(i).widget().setSelected(False)
        cord = event.pos().y()
        divider_index = next((i for i, x in enumerate(self.centers) if x > cord), len(self.centers))
        layerWidget = event.source()
        total = self.vbox_layout.count()//2 - 1
        index = total - self.vbox_layout.indexOf(layerWidget)//2 - 1
        insert = total - divider_index
        self.layers._insert_reorder(index, insert)
        event.accept()

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
