from napari._qt.widgets.qt_dims import QtDims, QtDimSliderWidget
from napari._qt.widgets.qt_dims_slider import SLIDER_MINIMUM_WIDTH
from napari.components import Dims


def test_same_margin_popup(qtbot):
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]
    # Lazily create the widget
    assert slider.margins_popup is None
    slider.show_margins_popup()
    old_margins_popup = slider.margins_popup
    assert old_margins_popup is not None
    # Reuse old margins popup
    slider.show_margins_popup()
    assert old_margins_popup is slider.margins_popup


def test_move_margin_popup(qtbot):
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]
    slider.show_margins_popup()
    # Check that values of the left slider matches the
    # values of the dims margin_right after the
    # margin_right has been moved within the dims
    dims.margin_right = (2, 0, 0)
    assert slider.margins_popup.right_slider.value() == dims.margin_right[0]
    slider.margins_popup.left_slider.setValue(1)
    assert slider.margins_popup.left_slider.value() == dims.margin_left[0]


def test_slider_has_a_minimum_width(qtbot):
    """The groove must not be squeezed to nothing by a narrow window.

    Without a minimum the row's minimum width comes only from its labels and
    the slider absorbs the whole shortfall, leaving a groove too short to
    position the handle on an axis with many steps.
    """
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    for slider_widget in view.slider_widgets:
        assert slider_widget.slider.minimumWidth() == SLIDER_MINIMUM_WIDTH


def test_slider_minimum_width_reaches_the_row(qtbot):
    """The constraint propagates, so a narrow window cannot collapse it."""
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider_widget: QtDimSliderWidget = view.slider_widgets[0]
    assert slider_widget.minimumSizeHint().width() >= SLIDER_MINIMUM_WIDTH
