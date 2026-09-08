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
from qtpy.QtGui import QColor
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
    QHeaderView,
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
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledSlider, QRangeSlider

import napari
from napari.utils.theme import available_themes, get_theme

# ── WCAG 2.1 helpers ───────────
_AA_NORMAL = 4.5
_AA_LARGE = 3.0
_AAA_NORMAL = 7.0
_AAA_LARGE = 4.5

_CONTRAST_PAIRS: list[tuple[str, str, str]] = [
    ('background', 'text', 'normal'),
    ('primary', 'text', 'normal'),
    ('foreground', 'text', 'normal'),
    ('current', 'text', 'normal'),
    ('highlight', 'text', 'normal'),
    ('console', 'text', 'normal'),
    ('background', 'icon', 'large'),
    ('primary', 'icon', 'large'),
    ('current', 'icon', 'large'),
]


def _linearize(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rel_lum(r: int, g: int, b: int) -> float:
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def _contrast_ratio(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    l1 = _rel_lum(*c1)
    l2 = _rel_lum(*c2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def _parse_rgb(color_str: str) -> tuple[int, int, int]:
    import ast
    if color_str.startswith('rgb('):
        return ast.literal_eval(color_str.lstrip('rgb(').rstrip(')'))
    named = {'black': (0, 0, 0), 'white': (255, 255, 255)}
    if color_str.lower() in named:
        return named[color_str.lower()]
    if color_str.startswith('#'):
        c = color_str.lstrip('#')
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    return (0, 0, 0)

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

        self.setMaximumHeight(88)


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
        self._theme_selector = QComboBox()
        self._current_button = QPushButton('current')
        self._theme_meta = QLabel()

        self.setMinimumWidth(800)
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
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)
        scroll_area.setWidget(content)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._build_theme_selector_group())
        left_layout.addWidget(self._build_state_group())
        left_layout.addWidget(self._build_controls_group())
        left_layout.addStretch(1)
        content_layout.addWidget(left_column, 2)

        right_column = QWidget()
        right_column.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self._build_color_group())
        self._contrast_group = self._build_contrast_group()
        right_layout.addWidget(self._contrast_group, 1)
        content_layout.addWidget(right_column, 3)

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
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(4)

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

        lab_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        lab_slider.setValue(10)
        layout.addWidget(lab_slider)

        h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        h_scrollbar.setValue(50)
        layout.addWidget(h_scrollbar)
        layout.addWidget(QRangeSlider(Qt.Orientation.Horizontal, self))

        text = QTextEdit()
        text.setMaximumHeight(70)
        text.setHtml(_BLURB)
        layout.addWidget(text)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(4)
        input_row.addWidget(QTimeEdit())

        edit = QLineEdit()
        edit.setPlaceholderText('LineEdit placeholder...')
        input_row.addWidget(edit, 1)
        layout.addLayout(input_row)

        progress = QProgressBar()
        progress.setValue(50)
        layout.addWidget(progress)

        radio_group = QGroupBox('Exclusive Radio Buttons')
        radio_layout = QHBoxLayout(radio_group)
        radio_layout.setContentsMargins(6, 6, 6, 6)
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

    def _build_contrast_group(self) -> QGroupBox:
        group = QGroupBox('Contrast ratios (WCAG 2.1)')
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 5, 6, 5)

        self._contrast_table = QTableWidget(len(_CONTRAST_PAIRS), 4)
        self._contrast_table.setHorizontalHeaderLabels(
            ['Pair', 'Ratio', 'AA', 'AAA']
        )
        self._contrast_table.verticalHeader().hide()
        self._contrast_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        header = self._contrast_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._contrast_table, 1)
        self._contrast_legend = QLabel(
            '<small>AA ≥ 4.5:1 (normal) / 3.0:1 (large) &middot; '
            'AAA ≥ 7.0:1 (normal) / 4.5:1 (large)</small>'
        )
        self._contrast_legend.setWordWrap(True)
        layout.addWidget(self._contrast_legend)
        return group

    def _refresh_contrast(self, theme_dict: dict) -> None:
        """Update the contrast table for the current theme."""
        parsed: dict[str, tuple[int, int, int]] = {}
        for k in ('background', 'primary', 'foreground', 'current',
                   'highlight', 'console', 'text', 'icon', 'canvas'):
            if k in theme_dict:
                parsed[k] = _parse_rgb(str(theme_dict[k]))

        # Filter to only WCAG pairs (skip visual-only ones with None aa)
        for row, (bg_key, fg_key, cat) in enumerate(_CONTRAST_PAIRS):
            if bg_key not in parsed or fg_key not in parsed:
                continue
            ratio = _contrast_ratio(parsed[bg_key], parsed[fg_key])
            threshold_aa = _AA_NORMAL if cat == 'normal' else _AA_LARGE
            threshold_aaa = _AAA_NORMAL if cat == 'normal' else _AAA_LARGE
            aa_pass = ratio >= threshold_aa
            aaa_pass = ratio >= threshold_aaa

            self._contrast_table.setItem(row, 0, QTableWidgetItem(f'{bg_key} / {fg_key}'))
            self._contrast_table.setItem(
                row, 1, QTableWidgetItem(f'{ratio:.1f}:1')
            )
            for col, ok in ((2, aa_pass), (3, aaa_pass)):
                item = QTableWidgetItem('Yes' if ok else 'No')
                if not ok:
                    item.setBackground(QColor(255, 220, 220))
                    item.setForeground(QColor(0, 0, 0))  # dark text on light bg
                self._contrast_table.setItem(row, col, item)

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
        self._refresh_contrast(theme_dict)


viewer = napari.Viewer(title='Theme sample', show_welcome_screen=False)
widget = ThemeSampleWidget(viewer)
dock_widget = viewer.window.add_dock_widget(
    widget,
    area='right',
    name='Theme sample',
)
dock_widget.setMinimumWidth(widget.minimumWidth())
viewer.open_sample('napari', 'cells3d')

if __name__ == '__main__':
    napari.run()
