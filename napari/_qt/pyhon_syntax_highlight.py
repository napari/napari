from pygments import highlight
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name
from pygments.styles import get_style_by_name
from qtpy import QtGui

# https://github.com/Vector35/snippets/blob/master/QCodeEditor.py


def _format(style):
    """Return a QTextCharFormat with the given attributes."""
    _format = QtGui.QTextCharFormat()
    if "#" in style:
        index = style.index("#")
        color = style[index : index + 7]
        _color = QtGui.QColor(color)

        _format.setForeground(_color)

    if "bold" in style and "nobold" not in style:
        _format.setFontWeight(QtGui.QFont.Bold)
    if "italic" in style and "noitalic" not in style:
        _format.setFontItalic(True)
    if "underline" in style and "nounderline" not in style:
        _format.setFontUnderline(True)

    return _format


class QFormatter(Formatter):
    def __init__(self, theme):
        super().__init__()
        self._theme = theme
        self._style = {
            str(name): _format(style)
            for name, style in get_style_by_name(self._theme).styles.items()
        }

    def format(self, tokensource, outfile):
        self.data = []

        for token, value in tokensource:
            self.data.extend(
                [
                    self._style[str(token)],
                ]
                * len(value)
            )


class Pylighter(QtGui.QSyntaxHighlighter):
    def __init__(self, parent, lang, theme):
        super().__init__(parent)
        self.formatter = QFormatter(theme)
        self.lexer = get_lexer_by_name(lang)

    def highlightBlock(self, text):
        cb = self.currentBlock()
        p = cb.position()
        text = self.document().toPlainText() + '\n'
        highlight(text, self.lexer, self.formatter)

        # dirty, dirty hack
        for i in range(len(text)):
            try:
                self.setFormat(i, 1, self.formatter.data[p + i])
            except IndexError:
                pass
