from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QSlider, QVBoxLayout, QSplitter



class QtViewer(QSplitter):
    def __init__(self, viewer):
        super().__init__()

        # To split vertical sliders, viewer and layerlist, minimumsizes given for demo purposes/NOT FINAL
        viewer.controlBars._qt.setMinimumSize(QSize(60, 60))
        self.addWidget(viewer.controlBars._qt)
        viewer.dimensions._qt.setMinimumSize(QSize(100, 100))
        self.addWidget(viewer.dimensions._qt)
        viewer.layers._qt.setMinimumSize(QSize(250, 250))
        self.addWidget(viewer.layers._qt)
