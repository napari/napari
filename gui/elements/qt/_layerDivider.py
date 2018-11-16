from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame

class QtDivider(QFrame):
    def __init__(self, name):
        super().__init__()
        self.unselectedStlyeSheet = "QFrame {border: 3px solid lightGray; background-color:lightGray; border-radius: 3px;}"
        self.selectedStlyeSheet = "QFrame {border: 3px solid rgb(71,143,205); background-color:rgb(71,143,205); border-radius: 3px;}"
        self.setStyleSheet(self.unselectedStlyeSheet)
        self.setFixedHeight(5)
        self.name = name
        self.setAcceptDrops(True)

    def dropEvent(self, event):
        print('Drop')
        self.setStyleSheet(self.unselectedStlyeSheet)
        #print(self.name + ' ' + event.mimeData().text())

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
