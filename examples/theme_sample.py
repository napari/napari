"""
Theme sample
============

Inspect napari themes with labeled color roles, common widget states, and a
live theme selector.

The sample widget is docked inside napari so it inherits the same stylesheet
used by the rest of the application. The theme selector includes builtin
themes and any plugin-contributed themes discovered when the viewer starts.

.. tags:: gui
"""

from __future__ import annotations

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

import napari
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

_COLOR_DESCRIPTIONS: dict[str, str] = {
    'canvas': 'Canvas or viewport background.',
    'console': 'Console or terminal background.',
    'background': 'Main application background.',
    'foreground': 'Layer controls and panel background.',
    'primary': 'Primary widget surfaces such as sliders and spin boxes.',
    'secondary': 'Secondary surfaces and text selection accents.',
    'highlight': 'Selection, hover, and highlighted states.',
    'text': 'Primary text and labels.',
    'icon': 'Icons and glyphs.',
    'warning': 'Warning indicators.',
    'error': 'Error indicators.',
    'current': 'Current or active layer accent color.',
}

class TabDemo(QTabWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        tab1 = QWidget()
        tab2 = QWidget()

        self.addTab(tab1, 'Tab 1')
        self.addTab(tab2, 'Tab 2')

        layout1 = QFormLayout(tab1)
        layout1.addRow('Height', QSpinBox())
        layout1.addRow('Weight', QDoubleSpinBox())

        layout2 = QFormLayout(tab2)
        sex_row = QHBoxLayout()
        sex_row.addWidget(QRadioButton('Male'))
        sex_row.addWidget(QRadioButton('Female'))
        layout2.addRow(QLabel('Sex'), sex_row)
        layout2.addRow('Date of Birth', QLineEdit())

        self.setMaximumHeight(100)


class ColorSwatch(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(26, 26)

    def set_color(self, color: str) -> None:
        self.setStyleSheet(
            f'background-color: {color}; border: 1px solid rgba(0, 0, 0, 0.25);'
        )


class ColorSwatchRow(QWidget):
    def __init__(
        self,
        role: str,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._role = role
        self._description = description
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._swatch = ColorSwatch()
        self._label = QLabel()
        self._label.setWordWrap(True)

        layout.addWidget(self._swatch)
        layout.addWidget(self._label, 1)

    def set_color(self, color: str) -> None:
        self._swatch.set_color(color)
        text = f'<b>{self._role}</b> &nbsp; {color}'
        if self._description:
            text += (
                f'<br/><span style="font-size: 9pt;">{self._description}</span>'
            )
        self._label.setText(text)

        tooltip = f'{self._role}: {color}'
        if self._description:
            tooltip += f'\n{self._description}'
        self.setToolTip(tooltip)
        self._label.setToolTip(tooltip)


class ThemeSampleWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._color_rows: dict[str, ColorSwatchRow] = {}
        self._header = QLabel()
        self._theme_selector = QComboBox()
        self._current_button = QPushButton('current')
        self._theme_meta = QLabel()

        self.setMinimumWidth(420)
        self._build_ui()
        self._viewer.events.theme.connect(self._sync_from_viewer_theme)
        self._sync_from_viewer_theme()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer_layout.addWidget(scroll_area)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(10)
        scroll_area.setWidget(content)

        self._header.setWordWrap(True)
        content_layout.addWidget(self._header)
        content_layout.addWidget(self._build_theme_selector_group())
        content_layout.addWidget(self._build_controls_group())
        content_layout.addWidget(self._build_state_group())
        content_layout.addWidget(self._build_color_group())
        content_layout.addStretch(1)

    def _build_theme_selector_group(self) -> QGroupBox:
        group = QGroupBox('Theme selector')
        layout = QFormLayout(group)

        self._theme_selector.addItems(available_themes())
        self._theme_selector.setToolTip(
            'Select a builtin or plugin-contributed napari theme.'
        )
        self._theme_selector.currentTextChanged.connect(
            self._on_theme_selected
        )
        layout.addRow('Theme', self._theme_selector)

        help_label = QLabel(
            'The selector lists builtin themes and any theme contributions '
            'discovered from installed plugins.'
        )
        help_label.setWordWrap(True)
        layout.addRow(help_label)
        return group

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox('Controls')
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        layout.addWidget(QPushButton('push button'))

        box = QComboBox()
        box.addItems(['a', 'b', 'c', 'cd'])
        layout.addWidget(box)
        layout.addWidget(QFontComboBox())

        checkbox_row = QHBoxLayout()
        partial = QCheckBox('tristate')
        partial.setTristate(True)
        partial.setCheckState(Qt.CheckState.PartiallyChecked)
        checked = QCheckBox('checked')
        checked.setChecked(True)
        checkbox_row.addWidget(QCheckBox('unchecked'))
        checkbox_row.addWidget(partial)
        checkbox_row.addWidget(checked)
        layout.addLayout(checkbox_row)

        layout.addWidget(TabDemo())

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setValue(50)
        layout.addWidget(slider)

        h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        h_scrollbar.setValue(50)
        layout.addWidget(h_scrollbar)
        layout.addWidget(QRangeSlider(Qt.Orientation.Horizontal, self))

        text = QTextEdit()
        text.setMaximumHeight(80)
        text.setHtml(_BLURB)
        layout.addWidget(text)
        layout.addWidget(QTimeEdit())

        edit = QLineEdit()
        edit.setPlaceholderText('LineEdit placeholder...')
        layout.addWidget(edit)

        progress = QProgressBar()
        progress.setValue(50)
        layout.addWidget(progress)

        radio_group = QGroupBox('Exclusive Radio Buttons')
        radio_layout = QHBoxLayout(radio_group)
        radio_layout.setContentsMargins(10, 10, 10, 10)
        radio1 = QRadioButton('&Radio button 1')
        radio2 = QRadioButton('R&adio button 2')
        radio3 = QRadioButton('Ra&dio button 3')
        radio1.setChecked(True)
        radio_layout.addWidget(radio1)
        radio_layout.addWidget(radio2)
        radio_layout.addWidget(radio3)
        radio_group.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        layout.addWidget(radio_group)
        return group

    def _build_state_group(self) -> QGroupBox:
        group = QGroupBox('Theme state samples')
        layout = QVBoxLayout(group)

        button_row = QHBoxLayout()
        button_row.addWidget(self._make_state_button('checked', checked=True))
        button_row.addWidget(self._make_state_button('unchecked'))
        button_row.addWidget(
            self._make_state_button('disabled', enabled=False)
        )
        self._current_button.setCheckable(True)
        self._current_button.setChecked(True)
        button_row.addWidget(self._current_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

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
        layout.addLayout(disabled_row)

        emphasized_frame = QFrame()
        emphasized_frame.setProperty('emphasized', True)
        emphasized_layout = QHBoxLayout(emphasized_frame)
        emphasized_layout.setContentsMargins(8, 6, 8, 6)
        emphasized_layout.addWidget(QLabel('Emphasized panel'))
        emphasized_layout.addStretch(1)
        layout.addWidget(emphasized_frame)

        return group

    def _build_color_group(self) -> QGroupBox:
        group = QGroupBox('Theme colors')
        layout = QVBoxLayout(group)

        grid = QGridLayout()
        grid.setSpacing(10)
        columns = 2

        for index, role in enumerate(_COLOR_DESCRIPTIONS):
            row_widget = ColorSwatchRow(role, _COLOR_DESCRIPTIONS[role])
            row, column = divmod(index, columns)
            grid.addWidget(row_widget, row, column)
            self._color_rows[role] = row_widget

        layout.addLayout(grid)
        layout.addWidget(self._theme_meta)
        return group

    def _make_state_button(
        self,
        label: str,
        *,
        checked: bool = False,
        enabled: bool = True,
    ) -> QPushButton:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setChecked(checked)
        button.setEnabled(enabled)
        return button

    def _on_theme_selected(self, theme_id: str) -> None:
        self._viewer.theme = theme_id

    def _sync_from_viewer_theme(self, event=None) -> None:
        theme_id = self._viewer.theme
        theme = get_theme(theme_id)
        resolved_id = theme.id

        if theme_id == 'system' and resolved_id != 'system':
            header = f'<h2>{theme.label} (system -&gt; {resolved_id})</h2>'
        else:
            header = f'<h2>{theme.label} ({theme_id})</h2>'
        self._header.setText(header)

        was_blocked = self._theme_selector.blockSignals(True)
        self._theme_selector.setCurrentText(theme_id)
        self._theme_selector.blockSignals(was_blocked)

        theme_dict = theme.to_rgb_dict()
        for role, row in self._color_rows.items():
            row.set_color(str(theme_dict[role]))

        self._theme_meta.setText(
            f'<b>Font size:</b> {theme_dict["font_size"]} '
            f'&nbsp;|&nbsp; <b>Syntax style:</b> {theme_dict["syntax_style"]}'
        )
        self._current_button.setStyleSheet(
            f'background-color: {theme.current.as_rgb()};'
        )


viewer = napari.Viewer(title='Theme sample', show_welcome_screen=False)
viewer.window._qt_window.resize(1400, 900)
widget = ThemeSampleWidget(viewer)
dock_widget = viewer.window.add_dock_widget(
    widget,
    area='right',
    name='Theme sample',
)

if __name__ == '__main__':
    napari.run()
