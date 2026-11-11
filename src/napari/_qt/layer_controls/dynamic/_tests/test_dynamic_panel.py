from itertools import combinations

from napari._qt.layer_controls.dynamic.buttons.qt_layer_buttons_base import (
    QtLayerButtons,
)
from napari._qt.layer_controls.dynamic.qt_dynamic_layer_controls import (
    QtDynamicLayerControls,
)
from napari._qt.layer_controls.dynamic.widgets import QtOpacityBlendingControls


def test_dynamic_controls_creation(layers, qtbot):
    for n_layers in range(1, 4):
        for selected_layers in combinations(layers, n_layers):
            callbacks = {lay: [] for lay in layers}
            for layer in layers:
                # reset opacity for next round
                layer.opacity = 1
                # count callbacks to make sure we don't leak
                callbacks[layer].extend(layer.events.callbacks)
                for emitter in layer.events.emitters.values():
                    callbacks[layer].extend(emitter.callbacks)

            controls = QtDynamicLayerControls(selected_layers)
            qtbot.addWidget(controls)

            if n_layers == 1:
                # buttons should show up
                buttons = controls.findChild(QtLayerButtons)
                assert buttons is not None
                buttons.ndisplay = 3
                assert buttons.ndisplay == 3

            # just test something runs
            opacity_slider = controls.findChild(
                QtOpacityBlendingControls
            ).opacity_slider
            opacity_slider.setValue(0.5)
            for layer in layers:
                if layer in selected_layers:
                    assert layer.opacity == 0.5
                else:
                    # make sure we don't affect other layers
                    assert layer.opacity == 1

            controls.close()
            # ensure that closing the controls removes all callbacks
            for layer in layers:
                new_callbacks = []
                new_callbacks.extend(layer.events.callbacks)
                for emitter in layer.events.emitters.values():
                    new_callbacks.extend(emitter.callbacks)
                assert set(new_callbacks) == set(callbacks[layer])
