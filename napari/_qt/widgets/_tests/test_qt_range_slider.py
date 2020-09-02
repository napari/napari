import os

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt

from napari._qt.widgets.qt_range_slider import QHRangeSlider, QVRangeSlider


@pytest.mark.parametrize('orientation', ['h', 'v'])
def test_range_slider(qtbot, orientation):
    model = QHRangeSlider if orientation == 'h' else QVRangeSlider
    initial = np.array([100, 400])
    range_ = np.array([0, 500])
    diff = abs(np.diff(range_))
    step = 1
    sld = model(initial_values=initial, data_range=range_, step_size=step)
    assert np.all([sld.value_min, sld.value_max] == (initial / diff))

    # test clicking parts triggers the right slider.moving
    assert sld.moving == 'none'
    if orientation == 'h':
        pos = sld.rangeSliderSize() * sld.value_min + sld.handle_radius
        pos = QPoint(int(pos), sld.height() // 2)
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos, delay=50)
        assert sld.moving == 'min'
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos - QPoint(50, 0), delay=50)
        assert sld.moving == 'min'
        pos = sld.rangeSliderSize() * sld.value_max + sld.handle_radius
        pos = QPoint(int(pos), sld.height() // 2)
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos, delay=50)
        assert sld.moving == 'max'
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos + QPoint(50, 0), delay=50)
        assert sld.moving == 'max'
        pos = sld.rangeSliderSize() * 0.5 + sld.handle_radius
        pos = QPoint(int(pos), sld.height() // 2)
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos, delay=50)
        assert sld.moving == 'bar'
    else:
        # for the vertical slider, somehow the required positions are changing
        # when the slider is not visible.  So we only test the vertical slider
        # on CI, to minimize GUI tests locally.
        if not os.getenv("CI"):
            return

        sld.show()
        pos = sld.rangeSliderSize() * sld.value_min + sld.handle_radius
        pos = QPoint(sld.width() // 2, int(pos))
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos, delay=50)
        assert sld.moving == 'max'
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos - QPoint(0, 50), delay=50)
        assert sld.moving == 'max'
        pos = sld.rangeSliderSize() * sld.value_max + sld.handle_radius
        pos = QPoint(sld.width() // 2, int(pos))
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos, delay=50)
        assert sld.moving == 'min'
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos + QPoint(0, 50), delay=50)
        assert sld.moving == 'min'
        pos = sld.rangeSliderSize() * 0.5 + sld.handle_radius
        pos = QPoint(sld.width() // 2, int(pos))
        qtbot.mousePress(sld, Qt.LeftButton, pos=pos, delay=50)
        assert sld.moving == 'bar'

    # can't simulate mouse moves... so directly set min and max and make sure
    # both the data values (sld.values()) and value_min/max update correctly
    sld.display_min = sld.handle_radius + 0.4 * sld.rangeSliderSize()
    sld.display_max = sld.handle_radius + 0.6 * sld.rangeSliderSize()
    sld.updateValuesFromDisplay()
    assert np.all([sld.value_min, sld.value_max] == [0.4, 0.6])
    assert np.all(sld.values() == np.array([0.4, 0.6]) * diff)

    # changing the range should update sld.values() but not sld.value_min/max
    sld.setRange(range_ * 2)
    assert np.all([sld.value_min, sld.value_max] == [0.4, 0.6])
    assert np.all(sld.values() == np.array([0.4, 0.6]) * diff * 2)

    qtbot.mouseRelease(sld, Qt.LeftButton, pos=pos, delay=50)
    assert sld.moving == 'none'

    # just make sure these don't crash for now
    sld.collapse()
    sld.expand()
