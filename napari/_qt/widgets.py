from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollBar,
    QSlider,
    QSpinBox,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QFormLayout,
)

from napari._qt.qt_range_slider import QHRangeSlider
from napari.resources import combine_stylesheets
from napari.utils.theme import palettes, template


raw_stylesheet = combine_stylesheets()

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
        self.setTabText(0, "Contact Details")
        self.tab1.setLayout(layout)

        layout2 = QFormLayout()
        sex = QHBoxLayout()
        sex.addWidget(QRadioButton("Male"))
        sex.addWidget(QRadioButton("Female"))
        layout2.addRow(QLabel("Sex"), sex)
        layout2.addRow("Date of Birth", QLineEdit())
        self.setTabText(1, "Personal Details")
        self.tab2.setLayout(layout2)

        self.setWindowTitle("tab demo")


class Widget(QWidget):
    def __init__(self, theme='dark', emphasized=False):
        super().__init__(None)
        self.setProperty('emphasized', emphasized)
        self.setStyleSheet(template(raw_stylesheet, **palettes[theme]))
        lay = QVBoxLayout()
        self.setLayout(lay)
        lay.addWidget(QPushButton('push button'))
        box = QComboBox()
        box.addItems(['a', 'b', 'c', 'cd'])
        lay.addWidget(box)
        lay.addWidget(QFontComboBox())
        chk = QCheckBox('check me (tristate)')
        chk.setToolTip('I am a tooltip')
        chk.setTristate(True)
        lay.addWidget(chk)
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
        dial = QDial()
        dial.setMaximumHeight(50)
        lay.addWidget(dial)
        lcd_num = QLCDNumber()
        lay.addWidget(lcd_num)
        dial.valueChanged.connect(lcd_num.display)


if __name__ == "__main__":

    app = QApplication([])
    w1 = Widget('dark')
    w1.setGeometry(200, 0, 425, 800)
    w1.show()
    w2 = Widget('dark', True)
    w2.setGeometry(700, 0, 425, 800)
    w2.show()
    app.exec_()
