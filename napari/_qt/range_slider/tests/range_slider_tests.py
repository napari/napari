"""
 Interactive test for range slider.
"""

# starts the QT event loop
from napari._qt.range_slider.range_slider import QVRangeSlider
from napari.util import app_context

with app_context():

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
