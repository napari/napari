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
def get_colorized_svg_xml(path: str, color: str = '#000000') -> str:
    """Return a colorized QImage from a path.  ``color`` must be valid css"""
    with open(path) as f:
        xml = f.read()
    css = svg_style_insert.format(color)
    xml = svg_tag_open.sub(f'\\1{css}', xml)
    return xml


class SVGIconEngine(QIconEngine):
    def __init__(self, xml) -> None:
        if isinstance(xml, str):
            xml = bytes(xml, encoding='utf-8')
        self.data = QByteArray(xml)
        super().__init__()

    def paint(self, painter, rect, mode, state):
        renderer = QSvgRenderer(self.data)
        renderer.render(painter, QRectF(rect))

    def clone(self):
        return SVGIconEngine(self.data)

    def pixmap(self, size, mode, state):
        img = QImage(size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        pix = QPixmap.fromImage(img, Qt.NoFormatConversion)
        painter = QPainter(pix)
        r = QRect(QPoint(0, 0), size)
        self.paint(painter, r, mode, state)
        return pix


@lru_cache()
def colored_svg_icon(*args):
    xml = get_colorized_svg_xml(*args)
    icon = QIcon(SVGIconEngine(xml))
    return icon
