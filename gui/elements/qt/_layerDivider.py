from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame

class QtDivider(QFrame):
    def __init__(self):
        super().__init__()
        self.unselectedStlyeSheet = "QFrame {border: 3px solid lightGray; background-color:lightGray; border-radius: 3px;}"
        self.selectedStlyeSheet = "QFrame {border: 3px solid rgb(71,143,205); background-color:rgb(71,143,205); border-radius: 3px;}"
        self.select(False)
        self.setFixedHeight(4)

    def select(self, bool):
        if bool:
            self.setStyleSheet(self.selectedStlyeSheet)
        else:
            self.setStyleSheet(self.unselectedStlyeSheet)
