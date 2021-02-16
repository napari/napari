from __future__ import annotations

import re
from functools import lru_cache

from qtpy.QtCore import QByteArray, QPoint, QRect, QRectF, Qt
from qtpy.QtGui import QIcon, QIconEngine, QImage, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer

svg_tag_open = re.compile(r'(<svg[^>]*>)')
svg_style_insert = """<style type="text/css">
path{{fill: {0}}}
polygon{{fill: {0}}}
circle{{fill: {0}}}
rect{{fill: {0}}}
</style>"""


@lru_cache()
def get_raw_svg(path: str) -> str:
    """Get and cached SVG XML.

    Raises
    ------
    ValueError
        If the path exists but does not contain valid SVG data.
    """
    with open(path) as f:
        xml = f.read()
        if not svg_tag_open.search(xml):
            raise ValueError(f"Could not detect svg tag in {path!r}")
        return xml


@lru_cache()
def get_colorized_svg(path_or_xml: str, color: str = '#000000') -> str:
    """Return a colorized version of the SVG XML at ``path``."""
    if '</svg>' in path_or_xml:
        xml = path_or_xml
    else:
        xml = get_raw_svg(path_or_xml)
    return svg_tag_open.sub(f'\\1{svg_style_insert.format(color)}', xml)


class QColoredSVGIcon(QIcon):
    """A QIcon class that specializes in colorizing SVG files.

    Parameters
    ----------
    path_or_xml : str
        Raw SVG XML or a path to an existing svg file.  (Will raise error on
        ``__init__`` if a non-existent file is provided.)
    color : str, optional
        A valid CSS color string, used to colorize the SVG. by default #000000
    """

    def __init__(self, path_or_xml: str, color: str = '#000000') -> None:
        xml = get_colorized_svg(path_or_xml, color)
        super().__init__(SVGBufferIconEngine(xml))

    @classmethod
    @lru_cache()
    def from_cache(cls, path_or_xml: str, color: str = '#000000'):
        """Create or get icon from cache if previously created.

        This is the preferred way to create a ``QColoredSVGIcon``.
        """
        return QColoredSVGIcon(path_or_xml, color)

    @classmethod
    def from_theme(cls, icon_name: str, theme: str, role: str = 'icon'):
        """Get an icon from resources using the ``role`` entry in ``theme``.

        Parameters
        ----------
        icon_name : str
            The name of the icon svg to load (just the stem).  Must be in the
            napari icons folder.
        theme : str
            Name of the theme to colorize icon with
        role : str, optional
            Key in the theme dict to use, by default 'icon'

        Returns
        -------
        QIcon
            A pre-colored QIcon
        """
        from ...resources import get_icon_path
        from ...utils.theme import get_theme

        color = get_theme(theme)[role]
        path = get_icon_path(icon_name)
        return QColoredSVGIcon.from_cache(path, color)


class SVGBufferIconEngine(QIconEngine):
    """A custom QIconEngine that can render an SVG buffer.

    An icon engine provides the rendering functions for a ``QIcon``.
    Each icon has a corresponding icon engine that is responsible for drawing
    the icon with a requested size, mode and state.  While the built-in
    QIconEngine is capable of rendering SVG files, it's not able to receive the
    raw XML string from memory.

    This ``QIconEngine`` takes in SVG data as a raw xml string or bytes.

    see: https://doc.qt.io/qt-5/qiconengine.html
    """

    def __init__(self, xml: str | bytes) -> None:
        if isinstance(xml, str):
            xml = bytes(xml, encoding='utf-8')
        self.data = QByteArray(xml)
        super().__init__()

    def paint(self, painter, rect, mode, state):
        """Paint the icon int ``rect`` using ``painter``."""
        renderer = QSvgRenderer(self.data)
        renderer.render(painter, QRectF(rect))

    def clone(self):
        """Required to subclass abstract QIconEngine."""
        return SVGBufferIconEngine(self.data)

    def pixmap(self, size, mode, state):
        """Return the icon as a pixmap with requested size, mode, and state."""
        img = QImage(size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        pixmap = QPixmap.fromImage(img, Qt.NoFormatConversion)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        return pixmap
