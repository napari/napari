import pytest
from qtpy.QtWidgets import QWidget

from napari._qt.utils import combine_widgets


class DummyWidget(QWidget):
    pass


def test_combine_widgets_error():
    """Check error raised when combining widgets with invalid types."""
    with pytest.raises(
        TypeError, match='"widgets" must be a QWidget, a magicgui'
    ):
        combine_widgets([DummyWidget(), 'string'])
