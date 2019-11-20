"""
An example of a rendering bug using QDockWidgets
"""

import sys
import numpy as np

from vispy import scene, plot

from qtpy.QtCore import Qt, QSize
from qtpy import QtWidgets


class QtPlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = scene.SceneCanvas(bgcolor='k', keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(300, 100))
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.canvas.native)
        _ = scene.visuals.Line(
            pos=np.array([[0, 0], [700, 500]]),
            color='w',
            parent=self.canvas.scene,
        )

        # self.plot = self.canvas.central_widget.add_widget(
        #     plot.PlotWidget(fg_color='w')
        # )
        # self.plot.histogram(np.random.randn(10000), bins=100, color='b')
        # the histogram is just an example, but the axes alone are sufficient:
        # self.plot._configure_2d()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(700, 500)
        dock_widget = QtWidgets.QDockWidget(self)
        self.plot = QtPlotWidget()
        dock_widget.setWidget(self.plot)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_widget)


if __name__ == '__main__':
    # import vispy
    # print(vispy.sys_info())
    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    appQt.exec_()
