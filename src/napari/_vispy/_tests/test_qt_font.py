import pytest
from qtpy.QtGui import QFont
from vispy.visuals.text.text import SDFRendererCPU

from napari._qt.utils import use_tabular_numerals
from napari._vispy.utils.qt_font import QtTextureFont


def test_digits_not_clipped_with_tabular_numerals(qapp):
    # napari enables tabular numerals on the app font. QPainter draws digits
    # with it, so they must also be measured with it or they get clipped.
    app_font = QFont(qapp.font())
    if not use_tabular_numerals(qapp):
        pytest.skip('tabular numerals need Qt >= 6.7')
    try:
        font = QtTextureFont(
            {'face': 'OpenSans', 'size': 12, 'bold': False, 'italic': False},
            SDFRendererCPU(),
        )
        for char in '0123456789':
            bitmap = font[char]['bitmap']
            # the image is padded, so ink in its last column means clipping
            assert not bitmap[:, -1].any(), char
    finally:
        qapp.setFont(app_font)
