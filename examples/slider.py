import sys
from PyQt5 import QtWidgets

from ..gui.util.range_slider import QHRangeSlider


# Slider Demo
app = QtWidgets.QApplication(sys.argv)
if True: # Toggle this to change between horizantal and vertical sliders
    hslider = QHRangeSlider(slider_range=[-15.0, 15.0, 0.02], values=[-2.5, 2.5])
    hslider.setEmitWhileMoving(True)
    hslider.show()
else:
    vslider = QVRangeSlider(slider_range=[-5.0, 5.0, 0.02], values=[-2.5, 2.5])
    vslider.setEmitWhileMoving(True)
    vslider.show()
sys.exit(app.exec_())
