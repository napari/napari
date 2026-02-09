"""SampleWidget that contains many types of QWidgets.

This file and SampleWidget is useful for testing out themes from the command
line or for generating screenshots of a sample widget to demonstrate a theme.
ThemeColorDisplay shows all theme colors as labeled swatches with hex values,
including derived colors (darken, lighten, opacity) used in the QSS.

Examples
--------
To use from the command line:

$ python -m napari._qt.widgets.qt_theme_sample

To include themes contributed by plugins:

$ python -m napari._qt.widgets.qt_theme_sample --include-plugins

To generate a screenshot within python:

>>> from napari._qt.widgets.qt_theme_sample import SampleWidget
>>> widg = SampleWidget(theme='dark')
>>> screenshot = widg.screenshot()

To view theme colors:

>>> from napari._qt.widgets.qt_theme_sample import ThemeColorDisplay
>>> widg = ThemeColorDisplay(theme='dark')
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


def _ensure_plugin_themes_loaded() -> bool:
    """Try to load plugin themes via the npe2 plugin manager."""
    try:
        from napari.plugins import _initialize_plugins
    except (ImportError, RuntimeError):
        return False
    _initialize_plugins()
    return True


class TabDemo(QTabWidget):
    def __init__(self, parent=None, emphasized=False) -> None:
        super().__init__(parent)
        self.setProperty('emphasized', emphasized)
        self.tab1 = QWidget()
        self.tab1.setProperty('emphasized', emphasized)
        self.tab2 = QWidget()
        self.tab2.setProperty('emphasized', emphasized)

        self.addTab(self.tab1, 'Tab 1')
        self.addTab(self.tab2, 'Tab 2')
        layout = QFormLayout()
        layout.addRow('Height', QSpinBox())
        layout.addRow('Weight', QDoubleSpinBox())
        self.setTabText(0, 'Tab 1')
        self.tab1.setLayout(layout)

        layout2 = QFormLayout()
        sex = QHBoxLayout()
        sex.addWidget(QRadioButton('Male'))
        sex.addWidget(QRadioButton('Female'))
        layout2.addRow(QLabel('Sex'), sex)
        layout2.addRow('Date of Birth', QLineEdit())
        self.setTabText(1, 'Tab 2')
        self.tab2.setLayout(layout2)

        self.setWindowTitle('tab demo')


class SampleWidget(QWidget):
    def __init__(self, theme='dark', emphasized=False) -> None:
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
        chk.setCheckState(Qt.CheckState.PartiallyChecked)
        chk3 = QCheckBox('checked')
        chk3.setChecked(True)
        hbox.addWidget(QCheckBox('unchecked'))
        hbox.addWidget(chk)
        hbox.addWidget(chk3)
        lay.addLayout(hbox)

        lay.addWidget(TabDemo(emphasized=emphasized))

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setValue(50)
        lay.addWidget(sld)
        scroll = QScrollBar(Qt.Orientation.Horizontal)
        scroll.setValue(50)
        lay.addWidget(scroll)
        lay.addWidget(QRangeSlider(Qt.Orientation.Horizontal, self))
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
        group_box = QGroupBox('Exclusive Radio Buttons')
        radio1 = QRadioButton('&Radio button 1')
        radio2 = QRadioButton('R&adio button 2')
        radio3 = QRadioButton('Ra&dio button 3')
        radio1.setChecked(True)
        hbox = QHBoxLayout()
        hbox.addWidget(radio1)
        hbox.addWidget(radio2)
        hbox.addWidget(radio3)
        hbox.addStretch(1)
        group_box.setLayout(hbox)
        lay.addWidget(group_box)

    def screenshot(self, path=None):
        img = self.grab().toImage()
        if path is not None:
            imsave(path, QImg2array(img))
        return QImg2array(img)


def _rgb_string_to_hex(rgb_string: str) -> str:
    """Convert rgb() or rgba() CSS string to hex."""
    if rgb_string.startswith('rgba('):
        parts = rgb_string.lstrip('rgba(').rstrip(')').split(',')
        r, g, b = (
            int(parts[0].strip()),
            int(parts[1].strip()),
            int(parts[2].strip()),
        )
        return f'#{r:02x}{g:02x}{b:02x}'
    if rgb_string.startswith('rgb('):
        parts = rgb_string.lstrip('rgb(').rstrip(')').split(',')
        r, g, b = (
            int(parts[0].strip()),
            int(parts[1].strip()),
            int(parts[2].strip()),
        )
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
    'current': 'Active/selected layer highlight (the blue accent)',
}


class ThemeColorDisplay(QWidget):
    """Display all theme colors as labeled swatches with hex values.

    Shows the base theme colors and commonly used derived colors
    (darken, lighten, opacity) that appear in the QSS stylesheets.

    Parameters
    ----------
    theme : str
        The napari theme id (e.g. 'dark', 'light').
    """

    def __init__(self, theme: str = 'dark') -> None:
        super().__init__(None)
        from napari.utils.theme import (
            darken,
            get_theme,
            lighten,
            opacity,
        )

        theme_obj = get_theme(theme)
        theme_dict = theme_obj.to_rgb_dict()

        # Main layout with title
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(20)

        # Title
        title_label = QLabel(f'<h2>{theme_obj.label} ({theme})</h2>')
        main_layout.addWidget(title_label)

        # --- Base colors section ---
        base_label = QLabel('<h3>Base Theme Colors</h3>')
        main_layout.addWidget(base_label)

        base_grid = QGridLayout()
        base_grid.setSpacing(10)

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
            row, col = divmod(i, 3)
            base_grid.addWidget(row_widget, row, col)

        main_layout.addLayout(base_grid)

        # --- Derived colors section ---
        derived_label = QLabel('<h3>Derived Colors (QSS functions)</h3>')
        main_layout.addWidget(derived_label)

        derived_grid = QGridLayout()
        derived_grid.setSpacing(10)

        # Gather the derived colors actually used in the QSS
        derived_colors: list[tuple[str, str, str]] = [
            (
                'darken(current, 10)',
                darken(theme_dict['current'], 10),
                'New points/shapes button background when checked',
            ),
            (
                'darken(foreground, 20)',
                darken(theme_dict['foreground'], 20),
                'Disabled mode radio buttons background',
            ),
            (
                'darken(background, 10)',
                darken(theme_dict['background'], 10),
                'Dock widget title bar hover state',
            ),
            (
                'lighten(error, 10)',
                lighten(theme_dict['error'], 10),
                'Play button background during recording',
            ),
            (
                'opacity(text, 90)',
                opacity(theme_dict['text'], 90),
                'Disabled widget text (slightly transparent)',
            ),
            (
                'lighten(foreground, 5)',
                lighten(theme_dict['foreground'], 5),
                'Subtle highlight on layer controls',
            ),
        ]

        for i, (role, color_val, desc) in enumerate(derived_colors):
            hex_color = _rgb_string_to_hex(color_val)
            row_widget = _make_swatch_row(role, hex_color, desc)
            row, col = divmod(i, 3)
            derived_grid.addWidget(row_widget, row, col)

        main_layout.addLayout(derived_grid)

        # --- Font info ---
        info_label = QLabel(
            f'<b>Font size:</b> {theme_dict.get("font_size", "?")} &nbsp;|&nbsp; '
            f'<b>Syntax style:</b> {theme_dict.get("syntax_style", "?")}'
        )
        main_layout.addWidget(info_label)

        main_layout.addStretch()

        # Use scroll area for overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        scroll_widget = QWidget()
        scroll_widget.setLayout(main_layout)
        scroll.setWidget(scroll_widget)

        # Outer layout
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        # Apply theme stylesheet to the whole widget
        self.setStyleSheet(get_stylesheet(theme))

    def screenshot(self, path=None):
        img = self.grab().toImage()
        if path is not None:
            imsave(path, QImg2array(img))
        return QImg2array(img)


if __name__ == '__main__':
    import argparse
    import logging

    from napari._qt.qt_event_loop import get_qapp
    from napari.utils.theme import available_themes

    parser = argparse.ArgumentParser(
        description='Show napari theme sample widgets and color swatches.'
    )
    parser.add_argument(
        'themes',
        nargs='*',
        help='Theme ids to display (default: all available themes).',
    )
    parser.add_argument(
        '--include-plugins',
        action='store_true',
        help='Discover npe2 plugin themes before listing available themes.',
    )
    args = parser.parse_args()

    if args.include_plugins:
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
        w.setGeometry(10 + 430 * n, 0, 425, 600)
        w.setWindowTitle(f'Widgets — {theme}')
        w.show()
        widgets.append(w)

        try:
            c = ThemeColorDisplay(theme)
        except KeyError:
            continue
        c.setGeometry(10 + 430 * n, 620, 600, 550)
        c.setWindowTitle(f'Theme Colors — {theme}')
        c.show()
        widgets.append(c)
    if widgets:
        app.exec_()
