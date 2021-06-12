"""SampleWidget that contains many types of QWidgets.

This file and SampleWidget is useful for testing out themes from the command
line or for generating screenshots of a sample widget to demonstrate a theme.

Examples
--------
To use from the command line:

$ python -m napari._qt.theme_sample

To generate a screenshot within python:

>>> from napari._qt.theme_sample import SampleWidget
>>> widg = SampleWidget(theme='dark')
>>> screenshot = widg.screenshot()
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollBar,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ...utils.io import imsave
from ..qt_resources import get_stylesheet
from ..utils import QImg2array
from .qt_range_slider import QHRangeSlider

blurb = """
<h3>Heading</h3>
<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation
ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur
sint occaecat cupidatat non proident, sunt in culpa qui
officia deserunt mollit anim id est laborum.</p>
"""


class TabDemo(QTabWidget):
    def __init__(self, parent=None, emphasized=False):
        super().__init__(parent)
        self.setProperty('emphasized', emphasized)
        self.tab1 = QWidget()
        self.tab1.setProperty('emphasized', emphasized)
        self.tab2 = QWidget()
        self.tab2.setProperty('emphasized', emphasized)

        self.addTab(self.tab1, "Tab 1")
        self.addTab(self.tab2, "Tab 2")
        layout = QFormLayout()
        layout.addRow("Height", QSpinBox())
        layout.addRow("Weight", QDoubleSpinBox())
        self.setTabText(0, "Tab 1")
        self.tab1.setLayout(layout)

        layout2 = QFormLayout()
        sex = QHBoxLayout()
        sex.addWidget(QRadioButton("Male"))
        sex.addWidget(QRadioButton("Female"))
        layout2.addRow(QLabel("Sex"), sex)
        layout2.addRow("Date of Birth", QLineEdit())
        self.setTabText(1, "Tab 2")
        self.tab2.setLayout(layout2)

        self.setWindowTitle("tab demo")


class SampleWidget(QWidget):
    def __init__(self, theme='dark', emphasized=False):
        super().__init__(None)
        self.setProperty('emphasized', emphasized)
        self.setStyleSheet(get_stylesheet(theme))
        lay = QVBoxLayout()
        self.setLayout(lay)
        lay.addWidget(QPushButton('push button'))
        box = QComboBox()
        box.addItems(['a', 'b', 'c', 'cd'])
        lay.addWidget(box)
        lay.addWidget(QFontComboBox())

        hbox = QHBoxLayout()
        chk = QCheckBox('tristate')
        chk.setToolTip('I am a tooltip')
        chk.setTristate(True)
        chk.setCheckState(Qt.PartiallyChecked)
        chk3 = QCheckBox('checked')
        chk3.setChecked(True)
        hbox.addWidget(QCheckBox('unchecked'))
        hbox.addWidget(chk)
        hbox.addWidget(chk3)
        lay.addLayout(hbox)

        lay.addWidget(TabDemo(emphasized=emphasized))

        sld = QSlider(Qt.Horizontal)
        sld.setValue(50)
        lay.addWidget(sld)
        scroll = QScrollBar(Qt.Horizontal)
        scroll.setValue(50)
        lay.addWidget(scroll)
        lay.addWidget(QHRangeSlider(parent=self))
        text = QTextEdit()
        text.setMaximumHeight(100)
        text.setHtml(blurb)
        lay.addWidget(text)
        lay.addWidget(QTimeEdit())
        edit = QLineEdit()
        edit.setPlaceholderText('LineEdit placeholder...')
        lay.addWidget(edit)
        lay.addWidget(QLabel('label'))
        prog = QProgressBar()
        prog.setValue(50)
        lay.addWidget(prog)
        groupBox = QGroupBox("Exclusive Radio Buttons")
        radio1 = QRadioButton("&Radio button 1")
        radio2 = QRadioButton("R&adio button 2")
        radio3 = QRadioButton("Ra&dio button 3")
        radio1.setChecked(True)
        hbox = QHBoxLayout()
        hbox.addWidget(radio1)
        hbox.addWidget(radio2)
        hbox.addWidget(radio3)
        hbox.addStretch(1)
        groupBox.setLayout(hbox)
        lay.addWidget(groupBox)

    def screenshot(self, path=None):
        img = self.grab().toImage()
        if path is not None:
            imsave(path, QImg2array(img))
        return QImg2array(img)


if __name__ == "__main__":
    import sys

    from ...utils.theme import available_themes
    from ..qt_event_loop import get_app

    themes = [sys.argv[1]] if len(sys.argv) > 1 else available_themes()
    app = get_app()
    widgets = []
    for n, theme in enumerate(themes):
        try:
            w = SampleWidget(theme)
        except KeyError:
            print(f"{theme} is not a recognized theme")
            continue
        w.setGeometry(10 + 430 * n, 0, 425, 600)
        w.show()
        widgets.append(w)
    if widgets:
        app.exec_()
