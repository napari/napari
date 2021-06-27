from qtpy.QtCore import QPoint, QRect, QSize, Qt
from qtpy.QtGui import QFontMetrics, QPainter, QTextLayout
from qtpy.QtWidgets import QFrame, QLabel, QWidget


class ElidingLabel(QLabel):
    """A single-line eliding QLabel."""

    def __init__(self, text='', parent=None):
        super().__init__(parent)
        self._text = text

    def setText(self, txt):
        self._text = txt
        short = self._getShortText(self.width())
        super().setText(short)

    def resizeEvent(self, rEvent):
        width = rEvent.size().width()
        short = self._getShortText(width)
        super().setText(short)
        rEvent.accept()

    def _getShortText(self, width):
        self.fm = QFontMetrics(self.font())
        short = self.fm.elidedText(self._text, Qt.ElideRight, width)
        return short


class MultilineElidedLabel(QFrame):
    """A multiline QLabel-like widget that elides the last line.

    Behaves like a multiline QLabel, but will fill the available vertical space
    set by the parent, and elide the last line of text (i.e. cut off with an
    ellipses.)

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget, by default None
    text : str, optional
        The text to show in the label, by default ''
    """

    def __init__(self, parent: QWidget = None, text: str = ''):
        super().__init__(parent)
        self.setText(text)

    def setText(self, text=None):
        if text is not None:
            self._text = text
        self.update()
        self.adjustSize()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        font_metrics = painter.fontMetrics()
        text_layout = QTextLayout(self._text, painter.font())
        text_layout.beginLayout()

        y = 0
        while True:
            line = text_layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(self.width())
            nextLineY = y + font_metrics.lineSpacing()
            if self.height() >= nextLineY + font_metrics.lineSpacing():
                line.draw(painter, QPoint(0, y))
                y = nextLineY
            else:
                lastLine = self._text[line.textStart() :]
                elidedLastLine = font_metrics.elidedText(
                    lastLine, Qt.ElideRight, self.width()
                )
                painter.drawText(
                    QPoint(0, y + font_metrics.ascent()), elidedLastLine
                )
                line = text_layout.createLine()
                break
        text_layout.endLayout()

    def sizeHint(self):
        font_metrics = QFontMetrics(self.font())
        r = font_metrics.boundingRect(
            QRect(QPoint(0, 0), self.size()),
            Qt.TextWordWrap | Qt.ElideRight,
            self._text,
        )
        return QSize(self.width(), r.height())
