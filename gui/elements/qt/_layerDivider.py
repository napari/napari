from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame
import weakref

class QtDivider(QFrame):
    def __init__(self, name, _qt_layer_list):
        super().__init__()
        self._qt_layer_list = weakref.proxy(_qt_layer_list)
        self.unselectedStlyeSheet = "QFrame {border: 3px solid lightGray; background-color:lightGray; border-radius: 3px;}"
        self.selectedStlyeSheet = "QFrame {border: 3px solid rgb(71,143,205); background-color:rgb(71,143,205); border-radius: 3px;}"
        self.setStyleSheet(self.unselectedStlyeSheet)
        self.setFixedHeight(5)
        self.name = name
        self.setAcceptDrops(True)

    def dropEvent(self, event):
        print('Drop')
        self.setStyleSheet(self.unselectedStlyeSheet)
        index = int(event.mimeData().data('index'))
        layerWidget = self._qt_layer_list.layersLayout.itemAt(index).widget()
        layers = layerWidget.layer.viewer.layers
        index = layers.index(layerWidget.layer)
        divider_index = self._qt_layer_list.layersLayout.indexOf(self)
        totalwidgets = self._qt_layer_list.layersLayout.count()
        insert_index = int(totalwidgets/2) - int(divider_index/2)-1
        print(index)
        print(insert_index)
        total = len(layers)
        list = [i for i in range(total)]
        list.pop(index)
        if insert_index <= index:
            list.insert(insert_index, index)
        else:
            list.insert(insert_index-1, index)
        print(list)
        layers.reorder(list)
        print('DONE')


    def dragEnterEvent(self, event):
        print('Accept')
        event.accept()
        self.setStyleSheet(self.selectedStlyeSheet)

    def dragLeaveEvent(self, event):
        print('Leave')
        event.ignore()
        self.setStyleSheet(self.unselectedStlyeSheet)
        # for others set unselected! self.setStyleSheet(self.unselectedStlyeSheet)
        # if only over neighbouring ones don't select any!!!!
        # deal with multiple simultaneously selected !!!!!!!!!!!
