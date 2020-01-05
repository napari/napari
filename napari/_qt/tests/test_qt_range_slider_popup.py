import numpy as np

from napari._qt.qt_range_slider_popup import QRangeSliderPopup


def test_range_slider_popup(qtbot):
    initial = np.array([100, 400])
    range_ = np.array([0, 500])
    precision = 2
    pop = QRangeSliderPopup(
        horizontal=True,
        precision=precision,
        initial_values=initial,
        data_range=range_,
        step_size=1,
    )
    pop.show()

    # make sure labels are correct
    assert float(pop.curmin_label.text()) == initial[0]
    assert float(pop.curmax_label.text()) == initial[1]
    assert np.all(pop.slider.range == range_)

    # make sure setting the slider updates the labels
    pop.slider.setValues((10, 20))
    assert float(pop.curmin_label.text()) == 10
    assert float(pop.curmax_label.text()) == 20
    # and vice versa
    pop.curmin_label.setText('100')
    pop.curmax_label.setText('300')
    pop.curmin_label.editingFinished.emit()
    pop.curmax_label.editingFinished.emit()
    assert np.all(pop.slider.values() == (100, 300))

    # make sure changing the range labels updates the slider
    pop.range_min_label.setText('100')
    pop.range_max_label.setText('1000')
    pop.range_min_label.editingFinished.emit()
    pop.range_max_label.editingFinished.emit()
    assert np.all(pop.slider.range == (100, 1000))
