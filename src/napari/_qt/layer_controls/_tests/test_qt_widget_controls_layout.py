import numpy as np
import pytest
from qtpy.QtWidgets import QFormLayout, QLabel, QPushButton

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari.layers import Image

ItemRole = QFormLayout.ItemRole


class _ControlWrapper:
    """Stand-in for a ``QtWidgetControlsBase`` subclass."""

    def __init__(self, rows):
        self._rows = rows

    def get_widget_controls(self):
        return self._rows


@pytest.fixture
def panel(qtbot):
    panel = QtLayerControls(Image(np.zeros((4, 4))))
    qtbot.addWidget(panel)
    return panel


def test_label_widget_row_keeps_its_label(panel):
    label = QLabel('label:')
    field = QPushButton('field')
    panel._add_widget_controls(_ControlWrapper([(label, field)]))

    layout = panel.layout()
    row, label_role = layout.getWidgetPosition(label)
    assert label_role == ItemRole.LabelRole
    assert layout.getWidgetPosition(field) == (row, ItemRole.FieldRole)
    assert layout.labelForField(field) is label


def test_single_widget_row_spans_form_without_label(panel):
    widget = QPushButton('full width')
    panel._add_widget_controls(_ControlWrapper([(widget,)]))

    layout = panel.layout()
    row, role = layout.getWidgetPosition(widget)
    assert role == ItemRole.SpanningRole
    assert layout.labelForField(widget) is None
    assert layout.itemAt(row, ItemRole.LabelRole) is None


def test_mixed_rows_are_laid_out_independently(panel):
    label = QLabel('label:')
    field = QPushButton('field')
    widget = QPushButton('full width')
    panel._add_widget_controls(_ControlWrapper([(label, field), (widget,)]))

    layout = panel.layout()
    label_row, _ = layout.getWidgetPosition(label)
    # rows are added in order: the single widget comes right after the pair
    assert layout.getWidgetPosition(widget) == (
        label_row + 1,
        ItemRole.SpanningRole,
    )
    assert layout.getWidgetPosition(field) == (label_row, ItemRole.FieldRole)
