"""SampleWidget that contains many types of QWidgets.

This file and SampleWidget is useful for testing out themes from the command
line or for generating screenshots of a sample widget to demonstrate a theme.
The sample widget includes labeled theme color swatches at the bottom.

Examples
--------
To use from the command line:

$ python -m napari._qt.widgets.qt_theme_sample

To generate a screenshot within python:

>>> from napari._qt.widgets.qt_theme_sample import SampleWidget
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
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from napari._qt.qt_resources import get_stylesheet
from napari._qt.utils import QImg2array
from napari.utils.io import imsave
from napari.utils.theme import available_themes, get_theme

_BLURB = """
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


def _ensure_plugin_themes_loaded() -> bool:
    """Try to load plugin themes via the npe2 plugin manager."""
    try:
        from napari.plugins import _initialize_plugins
    except (ImportError, RuntimeError):
        return False
    _initialize_plugins()
    return True


class TabDemo(QTabWidget):
    def __init__(
        self, parent: QWidget | None = None, emphasized=False
    ) -> None:
        super().__init__(parent)
        self.setProperty('emphasized', emphasized)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        for tab in (self.tab1, self.tab2):
            tab.setProperty('emphasized', emphasized)

        self.addTab(self.tab1, 'Tab 1')
        self.addTab(self.tab2, 'Tab 2')
        layout = QFormLayout(self.tab1)
        layout.addRow('Height', QSpinBox())
        layout.addRow('Weight', QDoubleSpinBox())

        layout2 = QFormLayout(self.tab2)
        sex_row = QHBoxLayout()
        sex_row.addWidget(QRadioButton('Male'))
        sex_row.addWidget(QRadioButton('Female'))
        layout2.addRow(QLabel('Sex'), sex_row)
        layout2.addRow('Date of Birth', QLineEdit())

        self.setWindowTitle('tab demo')
        self.setMaximumHeight(100)


def _build_theme_header(theme: str) -> QLabel:
    """Return a header label for the current theme."""
    theme_obj = get_theme(theme)
    return QLabel(f'<h2>{theme_obj.label} ({theme})</h2>')


class SampleWidget(QWidget):
    def __init__(self, theme='dark', emphasized=False) -> None:
        super().__init__(None)
        self.setProperty('emphasized', emphasized)
        self.setStyleSheet(get_stylesheet(theme))
        content = QWidget()
        content_layout = QVBoxLayout(content)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setWidget(content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll_area)
        content_layout.addWidget(_build_theme_header(theme))

        basics_group = QGroupBox('Controls')
        basics_layout = QVBoxLayout(basics_group)
        basics_layout.setContentsMargins(8, 6, 8, 6)
        basics_layout.setSpacing(6)
        basics_layout.addWidget(QPushButton('push button'))
        box = QComboBox()
        box.addItems(['a', 'b', 'c', 'cd'])
        basics_layout.addWidget(box)
        basics_layout.addWidget(QFontComboBox())

        hbox = QHBoxLayout()
        chk = QCheckBox('tristate')
        chk.setTristate(True)
        chk.setCheckState(Qt.CheckState.PartiallyChecked)
        chk3 = QCheckBox('checked')
        chk3.setChecked(True)
        hbox.addWidget(QCheckBox('unchecked'))
        hbox.addWidget(chk)
        hbox.addWidget(chk3)
        basics_layout.addLayout(hbox)
        basics_layout.addWidget(TabDemo(emphasized=emphasized))

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setValue(50)
        basics_layout.addWidget(sld)
        h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        h_scrollbar.setValue(50)
        basics_layout.addWidget(h_scrollbar)
        basics_layout.addWidget(QRangeSlider(Qt.Orientation.Horizontal, self))
        text = QTextEdit()
        text.setMaximumHeight(80)
        text.setHtml(_BLURB)
        basics_layout.addWidget(text)
        basics_layout.addWidget(QTimeEdit())
        edit = QLineEdit()
        edit.setPlaceholderText('LineEdit placeholder...')
        basics_layout.addWidget(edit)
        prog = QProgressBar()
        prog.setValue(50)
        basics_layout.addWidget(prog)
        group_box = QGroupBox('Exclusive Radio Buttons')
        radio1 = QRadioButton('&Radio button 1')
        radio2 = QRadioButton('R&adio button 2')
        radio3 = QRadioButton('Ra&dio button 3')
        radio1.setChecked(True)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(10, 10, 10, 10)
        hbox.addWidget(radio1)
        hbox.addWidget(radio2)
        hbox.addWidget(radio3)
        group_box.setLayout(hbox)
        group_box.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        basics_layout.addWidget(group_box)
        content_layout.addWidget(basics_group)

        # Theme-derived states and accents
        state_group = QGroupBox('Theme state samples')
        state_layout = QVBoxLayout(state_group)

        button_row = QHBoxLayout()
        button_row.addWidget(_make_state_button('checked', checked=True))
        button_row.addWidget(_make_state_button('unchecked'))
        button_row.addWidget(_make_state_button('disabled', enabled=False))
        button_row.addWidget(
            _make_state_button('current', checked=True, theme=theme)
        )
        button_row.addStretch(1)
        state_layout.addLayout(button_row)

        disabled_row = QHBoxLayout()
        disabled_label = QLabel('Disabled label')
        disabled_label.setEnabled(False)
        disabled_edit = QLineEdit('Disabled input')
        disabled_edit.setEnabled(False)
        disabled_check = QCheckBox('Disabled check')
        disabled_check.setEnabled(False)
        disabled_row.addWidget(disabled_label)
        disabled_row.addWidget(disabled_edit)
        disabled_row.addWidget(disabled_check)
        disabled_row.addStretch(1)
        state_layout.addLayout(disabled_row)

        emphasized_frame = QFrame()
        emphasized_frame.setProperty('emphasized', True)
        emphasized_layout = QHBoxLayout(emphasized_frame)
        emphasized_layout.setContentsMargins(8, 6, 8, 6)
        emphasized_layout.addWidget(QLabel('Emphasized panel'))
        emphasized_layout.addStretch(1)
        state_layout.addWidget(emphasized_frame)

        content_layout.addWidget(state_group)

        content_layout.addWidget(_build_color_swatches(theme))

    def screenshot(self, path=None):
        img = self.grab().toImage()
        if path is not None:
            imsave(path, QImg2array(img))
        return QImg2array(img)


def _rgb_string_to_hex(rgb_string: str) -> str:
    """Convert rgb() or rgba() CSS string to hex."""
    if rgb_string.startswith('rgb'):
        parts = rgb_string[rgb_string.find('(') + 1 : -1].split(',')
        if len(parts) >= 3:
            r, g, b = (int(p.strip()) for p in parts[:3])
            return f'#{r:02x}{g:02x}{b:02x}'
    # Already hex or named color
    return rgb_string


class ColorSwatch(QFrame):
    """A single color swatch box."""

    def __init__(
        self,
        hex_color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('colorSwatch')
        self.setFixedSize(26, 26)
        self.setStyleSheet(
            f'#colorSwatch {{ background: {hex_color}; border: 1px solid #888; border-radius: 3px; }}'
        )


def _make_swatch_row(
    role: str,
    hex_color: str,
    description: str,
) -> QWidget:
    """Create a row widget with a color swatch and label text."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    swatch = ColorSwatch(hex_color)
    label_text = f'<b>{role}</b> &nbsp; {hex_color}'
    if description:
        label_text += (
            f'<br/><span style="font-size: 9pt;">{description}</span>'
        )

    label = QLabel(label_text)
    label.setWordWrap(True)

    tooltip = f'{role}: {hex_color}'
    if description:
        tooltip += f'\n{description}'
    container.setToolTip(tooltip)
    label.setToolTip(tooltip)

    layout.addWidget(swatch)
    layout.addWidget(label, 1)
    return container


def _make_state_button(
    label: str,
    *,
    checked: bool = False,
    enabled: bool = True,
    theme: str | None = None,
) -> QPushButton:
    """Create a minimal themed button for QSS state display."""
    button = QPushButton(label)
    button.setCheckable(True)
    button.setChecked(checked)
    button.setEnabled(enabled)
    if label == 'current' and theme is not None:
        button.setStyleSheet(
            f'background-color: {get_theme(theme).current.as_rgb()};'
        )
    return button


def _build_color_swatches(theme: str) -> QGroupBox:
    """Create a group box with labeled theme color swatches."""
    theme_obj = get_theme(theme)
    theme_dict = theme_obj.to_rgb_dict()

    group = QGroupBox('Theme colors')
    layout = QVBoxLayout()
    group.setLayout(layout)

    grid = QGridLayout()
    grid.setSpacing(10)
    columns = 2

    color_roles = [
        'canvas',
        'console',
        'background',
        'foreground',
        'primary',
        'secondary',
        'highlight',
        'text',
        'icon',
        'warning',
        'error',
        'current',
    ]

    for i, role in enumerate(color_roles):
        color_val = theme_dict.get(role, '')
        hex_color = _rgb_string_to_hex(str(color_val))
        desc = _COLOR_DESCRIPTIONS.get(role, '')
        row_widget = _make_swatch_row(role, hex_color, desc)
        row, col = divmod(i, columns)
        grid.addWidget(row_widget, row, col)

    layout.addLayout(grid)

    info_label = QLabel(
        f'<b>Font size:</b> {theme_dict.get("font_size", "?")} &nbsp;|&nbsp; '
        f'<b>Syntax style:</b> {theme_dict.get("syntax_style", "?")}'
    )
    layout.addWidget(info_label)
    return group


# Descriptions for each theme color role
_COLOR_DESCRIPTIONS: dict[str, str] = {
    'canvas': 'Canvas/viewport background (the main viewing area)',
    'console': 'Console/terminal background',
    'background': 'Main application background (preferences, dialogs)',
    'foreground': 'Layer controls panel background',
    'primary': 'Layer control widgets (sliders, spinboxes) background',
    'secondary': 'Text selection background, secondary UI elements',
    'highlight': 'Checked buttons, active selections, hover states',
    'text': 'Primary text and labels throughout the UI',
    'icon': 'Icon and button glyph colors',
    'warning': 'Warning messages and indicators',
    'error': 'Error messages and critical indicators',
    'current': 'Active/selected layer highlight',
}


if __name__ == '__main__':
    import argparse
    import logging

    from napari._qt.qt_event_loop import get_qapp

    parser = argparse.ArgumentParser(
        description='Show napari theme sample widgets and color swatches.'
    )
    parser.add_argument(
        'themes',
        nargs='*',
        help='Theme ids to display (default: all available themes).',
    )
    args = parser.parse_args()

    _ensure_plugin_themes_loaded()

    themes = args.themes if args.themes else available_themes()
    app = get_qapp()
    widgets = []
    for n, theme in enumerate(themes):
        try:
            w = SampleWidget(theme)
        except KeyError:
            logging.getLogger('napari').warning(
                '%s is not a recognized theme', theme
            )
            continue
        w.adjustSize()
        screen = app.primaryScreen()
        max_height = (
            screen.availableGeometry().height() - 100 if screen else 900
        )
        w.setMaximumHeight(max_height)
        w.setGeometry(10 + 520 * n, 0, 520, max_height)
        w.setWindowTitle(f'Widgets — {theme}')
        w.show()
        widgets.append(w)
    if widgets:
        app.exec_()
