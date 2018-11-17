from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame
import weakref

class QtDivider(QFrame):
    def __init__(self, name, _qt_layer_list):
        super().__init__()
        self._qt_layer_list = weakref.proxy(_qt_layer_list)
        self.unselectedStlyeSheet = "QFrame {border: 3px solid lightGray; background-color:lightGray; border-radius: 3px;}"
        self.selectedStlyeSheet = "QFrame {border: 3px solid rgb(71,143,205); background-color:rgb(71,143,205); border-radius: 3px;}"
        self.select(False)
        self.setFixedHeight(5)
        self.name = name
        self.setAcceptDrops(True)

    def select(self, bool):
        if bool:
            self.setStyleSheet(self.selectedStlyeSheet)
        else:
            self.setStyleSheet(self.unselectedStlyeSheet)

    def dropEvent(self, event):
        print('Drop')
        self.select(False)
        index = int(event.mimeData().data('index'))
        layerWidget = self._qt_layer_list.layersLayout.itemAt(index).widget()
        layers = layerWidget.layer.viewer.layers
        index = layers.index(layerWidget.layer)
        divider_index = self._qt_layer_list.layersLayout.indexOf(self)
        totalwidgets = self._qt_layer_list.layersLayout.count()
        insert_index = int(totalwidgets/2) - int(divider_index/2)-1
        total = len(layers)
        list = [i for i in range(total)]
        list.pop(index)
        if insert_index <= index:
            list.insert(insert_index, index)
        else:
            list.insert(insert_index-1, index)
        layers.reorder(list)
        print('DONE')


    def dragEnterEvent(self, event):
        print('Accept')
        event.accept()
        index = int(event.mimeData().data('index'))
        layerWidget = self._qt_layer_list.layersLayout.itemAt(index).widget()
        layers = layerWidget.layer.viewer.layers
        index = layers.index(layerWidget.layer)
        divider_index = self._qt_layer_list.layersLayout.indexOf(self)
        totalwidgets = self._qt_layer_list.layersLayout.count()
        insert_index = int(totalwidgets/2) - int(divider_index/2)-1
        print(str(index) + str(insert_index))
        if not (insert_index == index) and not (insert_index-1 == index):
            self.select(True)

    def dragLeaveEvent(self, event):
        print('Leave')
        #event.ignore()
        #self.setStyleSheet(self.unselectedStlyeSheet)
