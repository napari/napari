from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame

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
