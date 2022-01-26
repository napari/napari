import numpy as np

from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari.layers import Points


def test_n_dimensional_checkbox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Points(np.random.rand(10, 2))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.ndimCheckBox

    assert layer.n_dimensional is False
    combo.setChecked(True)
    assert layer.n_dimensional is True
