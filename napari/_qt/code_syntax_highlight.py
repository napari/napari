from pygments import highlight
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name
from qtpy import QtGui

# inspired by  https://github.com/Vector35/snippets/blob/master/QCodeEditor.py (MIT license) and
# https://pygments.org/docs/formatterdevelopment/#html-3-2-formatter


def get_text_char_format(style):
    """
    Return a QTextCharFormat with the given attributes.

    https://pygments.org/docs/formatterdevelopment/#html-3-2-formatter
    """

    text_char_format = QtGui.QTextCharFormat()
    try:
        text_char_format.setFontFamilies(["monospace"])
    except AttributeError:
        text_char_format.setFontFamily(
            "monospace"
        )  # backward compatibility for pyqt5 5.12.3
    if style.get('color'):
        text_char_format.setForeground(QtGui.QColor(f"#{style['color']}"))

    if style.get('bgcolor'):
        text_char_format.setBackground(QtGui.QColor(style['bgcolor']))

    if style.get('bold'):
        text_char_format.setFontWeight(QtGui.QFont.Bold)
    if style.get('italic'):
        text_char_format.setFontItalic(True)
    if style.get("underline"):
        text_char_format.setFontUnderline(True)

    # TODO find if it is possible to support border style.

    return text_char_format


class QFormatter(Formatter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = []
        self._style = {
            name: get_text_char_format(style) for name, style in self.style
        }

    def format(self, tokensource, outfile):
        """
        `outfile` is argument from parent class, but
        in Qt we do not produce string output, but QTextCharFormat, so it needs to be
        collected using `self.data`.
        """
        self.data = []

        for token, value in tokensource:
            self.data.extend(
                [
                    self._style[token],
                ]
                * len(value)
            )


class Pylighter(QtGui.QSyntaxHighlighter):
    def __init__(self, parent, lang, theme) -> None:
        super().__init__(parent)
        self.formatter = QFormatter(style=theme)
        self.lexer = get_lexer_by_name(lang)

    def highlightBlock(self, text):
        cb = self.currentBlock()
        p = cb.position()
        text = self.document().toPlainText() + '\n'
        highlight(text, self.lexer, self.formatter)

        # dirty, dirty hack
        # The core problem is that pygemnts by default use string streams,
        # that will not handle QTextCharFormat, so wee need use `data` property to work around this.
        for i in range(len(text)):
            try:
                self.setFormat(i, 1, self.formatter.data[p + i])
            except IndexError:
                pass
