import pytest
from qtpy.QtGui import QFont, QFontMetricsF

from napari._vispy.utils.qt_font import _load_glyph_qt


@pytest.mark.usefixtures('qapp')
def test_glyph_image_covers_advance():
    # '1' has large side bearings, so its ink box is much narrower than its
    # advance; a narrow glyph image rendered it with the stem missing
    qfont = QFont('Open Sans', 256)
    metrics = QFontMetricsF(qfont)
    glyphs: dict[str, dict] = {}
    _load_glyph_qt(qfont, metrics, '1', glyphs)
    assert glyphs['1']['bitmap'].shape[1] >= metrics.horizontalAdvance('1')
