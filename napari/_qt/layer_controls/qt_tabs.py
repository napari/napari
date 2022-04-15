from qtpy.QtGui import QTransform
from qtpy.QtWidgets import QLabel, QTabBar, QTabWidget

from ..qt_resources import QColoredSVGIcon


class QtTabsWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__()

    def addTabs(self, tabdict):

        cnt = 0
        for tab, value in tabdict.items():
            self.addTab(value['widget'], "")
            # get the right icon.
            icon = QColoredSVGIcon.from_resources(value['icon'])
            icon = icon.colored(color='#FFFFFF').pixmap(20, 20)
            trans = QTransform()
            icon.transformed(trans.rotate(90))
            label = QLabel()
            label.setPixmap(icon)

            self.tabBar().setTabButton(cnt, QTabBar.LeftSide, label)
            cnt += 1
        self.setTabPosition(QTabWidget.West)
        self.setTabShape(QTabWidget.Rounded)
