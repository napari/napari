import sys
from PyQt5.QtWidgets import QApplication, QWidget

from gui.components import Dims
from gui.components._dims.view import QtDims


app = QApplication(sys.argv)

dims = Dims()

widget = QtDims(dims)
widget.show()

# Start the event loop.

sys.exit(app.exec())
