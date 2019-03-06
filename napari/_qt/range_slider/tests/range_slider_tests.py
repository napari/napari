"""
 Interactive test for range slider.
"""

import sys

from PyQt5.QtWidgets import QApplication

# starts the QT event loop
from napari._qt.range_slider.range_slider import QVRangeSlider

app = QApplication(sys.argv)

# creates a slider range widget::
widget = QVRangeSlider(slider_range=[0, 100, 1],
                       values=[0, 50],
                       parent=None)

# notify of changes while sliding:
widget.setEmitWhileMoving(True)

# allows range slider to collapse to a single knob:
widget.collapsable = True


# listener for change events:
def slider_changed(slidermin, slidermax):
    print((slidermin, slidermax))


# linking the listener to the slider:
widget.rangeChanged.connect(slider_changed)

# makes the widget visible on the desktop:
widget.show()

# Start the QT event loop.
sys.exit(app.exec())
