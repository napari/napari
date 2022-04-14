from qtpy.QtGui import QPixmap, QTransform
from qtpy.QtWidgets import QLabel, QTabBar, QTabWidget


class QtTabsWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__()

    def addTabs(self, tabdict):

        cnt = 0
        for tab, value in tabdict.items():
            self.addTab(value['widget'], "")
            # get the right icon.
            pm = QPixmap(value['icon'])
            pm = pm.scaled(20, 20)
            trans = QTransform()
            pm.transformed(trans.rotate(90))
            label = QLabel()
            label.setPixmap(pm)

            self.tabBar().setTabButton(cnt, QTabBar.LeftSide, label)
            cnt += 1
        self.setTabPosition(QTabWidget.West)
        self.setTabShape(QTabWidget.Rounded)
