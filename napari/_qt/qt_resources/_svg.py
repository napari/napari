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
def get_raw_svg(path: str):
    """Get and cached svg xml."""
    with open(path) as f:
        xml = f.read()
        if not svg_tag_open.search(xml):
            raise ValueError(f"Could not detect svg tag in {path!r}")
        return xml


@lru_cache()
def get_colorized_svg(path_or_xml: str, color: str = '#000000') -> str:
    """Return a colorized version of the SVG at ``path``."""
    if '</svg>' in path_or_xml:
        xml = path_or_xml
    else:
        xml = get_raw_svg(path_or_xml)
    return svg_tag_open.sub(f'\\1{svg_style_insert.format(color)}', xml)


class QColoredSVGIcon(QIcon):
    def __init__(self, path_or_xml: str, color: str = '#000000') -> None:
        xml = get_colorized_svg(path_or_xml, color)
        super().__init__(SVGBufferIconEngine(xml))

    @classmethod
    def from_theme(cls, theme: str, icon_name: str, role: str = 'icon'):
        from ...resources import get_icon_path
        from ...utils.theme import get_theme

        color = get_theme(theme)[role]
        path = get_icon_path(icon_name)
        return QColoredSVGIcon.from_cache(path, color)

    @classmethod
    @lru_cache()
    def from_cache(cls, *args):
        return QColoredSVGIcon(*args)


class SVGBufferIconEngine(QIconEngine):
    def __init__(self, xml) -> None:
        if isinstance(xml, str):
            xml = bytes(xml, encoding='utf-8')
        self.data = QByteArray(xml)
        super().__init__()

    def paint(self, painter, rect, mode, state):
        renderer = QSvgRenderer(self.data)
        renderer.render(painter, QRectF(rect))

    def clone(self):
        return SVGBufferIconEngine(self.data)

    def pixmap(self, size, mode, state):
        img = QImage(size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        pix = QPixmap.fromImage(img, Qt.NoFormatConversion)
        painter = QPainter(pix)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        return pix


# def show_path(path, color=None):
#     from qtpy.QtWidgets import QToolButton
#     from qtpy.QtCore import QSize

#     icon = QColoredSVGIcon.from_cache(path, color)
#     btn = QToolButton()
#     btn.setIcon(icon)
#     btn.setIconSize(QSize(300, 300))
#     btn.show()
#     return btn


# def show_icon(theme: str, icon_name: str, role: str = 'icon'):
#     from ...utils.theme import get_theme
#     from ...resources import get_icon_path

#     color = get_theme(theme)[role]
#     path = get_icon_path(icon_name)
#     return show_path(path, color)
