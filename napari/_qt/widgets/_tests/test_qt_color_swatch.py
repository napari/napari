import pytest
from qtpy.QtWidgets import QWidget

from napari._qt.widgets.qt_color_swatch import QColorSwatch, QColorSwatchEdit


@pytest.mark.parametrize('parent', [None, True])
@pytest.mark.parametrize('color', [None, [1, 1, 1, 1]])
@pytest.mark.parametrize('tooltip', [None, 'This is a test'])
def test_succesfull_create_qcolorswatchedit(qtbot, parent, color, tooltip):
    if parent:
        parent = QWidget()
        qtbot.add_widget(parent)
    qtbot.add_widget(
        QColorSwatchEdit(parent, initial_color=color, tooltip=tooltip)
    )


@pytest.mark.parametrize('parent', [None, True])
@pytest.mark.parametrize('color', [None, [1, 1, 1, 1]])
@pytest.mark.parametrize('tooltip', [None, 'This is a test'])
def test_succesfull_create_qcolorswatch(qtbot, parent, color, tooltip):
    if parent:
        parent = QWidget()
        qtbot.add_widget(parent)
    qtbot.add_widget(
        QColorSwatch(parent, initial_color=color, tooltip=tooltip)
    )
