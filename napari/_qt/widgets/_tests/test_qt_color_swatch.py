import itertools

from qtpy.QtWidgets import QWidget

from napari._qt.widgets.qt_color_swatch import QColorSwatch, QColorSwatchEdit


def test_succesfull_create_qcolorswatchedit(qtbot):
    parents = [
        None,
        QWidget(),
    ]
    colors = [
        None,
        [1, 1, 1, 1],
    ]
    tooltips = [
        None,
        'This is a test',
    ]

    for parent, color, tooltip in itertools.product(parents, colors, tooltips):
        QColorSwatchEdit(parent, initial_color=color, tooltip=tooltip)


def test_succesfull_create_qcolorswatch(qtbot):
    parents = [
        None,
        QWidget(),
    ]
    colors = [
        None,
        [1, 1, 1, 1],
    ]
    tooltips = [
        None,
        'This is a test',
    ]

    for parent, color, tooltip in itertools.product(parents, colors, tooltips):
        QColorSwatch(parent, initial_color=color, tooltip=tooltip)
