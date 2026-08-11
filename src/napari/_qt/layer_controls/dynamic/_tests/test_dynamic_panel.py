from itertools import combinations

from napari._qt.layer_controls.dynamic.qt_dynamic_layer_controls import (
    QtDynamicLayerControls,
)
from napari._qt.layer_controls.dynamic.widgets import QtOpacityBlendingControls


def test_dynamic_controls_creation(layers, qtbot):
    for n_layers in range(4):
        for selected_layers in combinations(layers, n_layers):
            # reset opacity for next round
            for layer in layers:
                layer.opacity = 1
            controls = QtDynamicLayerControls(layers)
            qtbot.addWidget(controls)

            # just test something runs
            opacity_slider = controls.findChild(
                QtOpacityBlendingControls
            ).opacity_slider
            opacity_slider.setValue(0.5)
            for layer in selected_layers:
                assert layer.opacity == 0.5
