from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QSlider,
)

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import (
    qt_signals_blocked,
    set_widgets_enabled_with_opacity,
)
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton
from napari.layers.base._base_constants import Mode
from napari.utils.action_manager import action_manager
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtTracksControls(QtLayerControls):
    """Qt view and controls for the Tracks layer.

    Parameters
    ----------
    layer : napari.layers.Tracks
        An instance of a Tracks layer.

    Attributes
    ----------
    layer : layers.Tracks
        An instance of a Tracks layer.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button for pan/zoom mode.
    transform_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select transform mode.

    """

    layer: 'napari.layers.Tracks'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # NOTE(arl): there are no events fired for changing checkboxes
        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.editable.connect(self._on_editable_or_visible_change)
        self.layer.events.visible.connect(self._on_editable_or_visible_change)
        self.layer.events.color_by.connect(self._on_color_by_change)
        self.layer.events.tail_width.connect(self._on_tail_width_change)
        self.layer.events.tail_length.connect(self._on_tail_length_change)
        self.layer.events.head_length.connect(self._on_head_length_change)
        self.layer.events.properties.connect(self._on_properties_change)
        self.layer.events.colormap.connect(self._on_colormap_change)

        def _radio_button(
            parent,
            btn_name,
            mode,
            action_name,
            extra_tooltip_text='',
            **kwargs,
        ):
            """
            Convenience local function to create a RadioButton and bind it to
            an action at the same time.

            Parameters
            ----------
            parent : Any
                Parent of the generated QtModeRadioButton
            btn_name : str
                name fo the button
            mode : Enum
                Value Associated to current button
            action_name : str
                Action triggered when button pressed
            extra_tooltip_text : str
                Text you want added after the automatic tooltip set by the
                action manager
            **kwargs:
                Passed to QtModeRadioButton

            Returns
            -------
            button: QtModeRadioButton
                button bound (or that will be bound to) to action `action_name`

            Notes
            -----
            When shortcuts are modifed/added/removed via the action manager, the
            tooltip will be updated to reflect the new shortcut.
            """
            action_name = f'napari:{action_name}'
            btn = QtModeRadioButton(parent, btn_name, mode, **kwargs)
            action_manager.bind_button(
                action_name,
                btn,
                extra_tooltip_text='',
            )
            return btn

        self.panzoom_button = _radio_button(
            layer,
            'pan_zoom',
            Mode.PAN_ZOOM,
            'activate_tracks_pan_zoom_mode',
            extra_tooltip_text=trans._('(or hold Space)'),
            checked=True,
        )
        self.transform_button = _radio_button(
            layer, 'pan', Mode.TRANSFORM, 'activate_tracks_transform_mode'
        )
        self._EDIT_BUTTONS = (self.transform_button,)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.transform_button)

        button_grid = QGridLayout()
        button_grid.addWidget(self.panzoom_button, 0, 6)
        button_grid.addWidget(self.transform_button, 0, 7)
        button_grid.setContentsMargins(5, 0, 0, 5)
        button_grid.setColumnStretch(0, 1)
        button_grid.setSpacing(4)

        # combo box for track coloring, we can get these from the properties
        # keys
        self.color_by_combobox = QComboBox()
        self.color_by_combobox.addItems(self.layer.properties_to_color_by)

        self.colormap_combobox = QComboBox()
        for name, colormap in AVAILABLE_COLORMAPS.items():
            display_name = colormap._display_name
            self.colormap_combobox.addItem(display_name, name)

        # slider for track head length
        self.head_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.head_length_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.head_length_slider.setMinimum(0)
        self.head_length_slider.setMaximum(self.layer._max_length)
        self.head_length_slider.setSingleStep(1)

        # slider for track tail length
        self.tail_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.tail_length_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tail_length_slider.setMinimum(1)
        self.tail_length_slider.setMaximum(self.layer._max_length)
        self.tail_length_slider.setSingleStep(1)

        # slider for track edge width
        self.tail_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.tail_width_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tail_width_slider.setMinimum(1)
        self.tail_width_slider.setMaximum(int(2 * self.layer._max_width))
        self.tail_width_slider.setSingleStep(1)

        # checkboxes for display
        self.id_checkbox = QCheckBox()
        self.tail_checkbox = QCheckBox()
        self.tail_checkbox.setChecked(True)
        self.graph_checkbox = QCheckBox()
        self.graph_checkbox.setChecked(True)

        self.tail_width_slider.valueChanged.connect(self.change_tail_width)
        self.tail_length_slider.valueChanged.connect(self.change_tail_length)
        self.head_length_slider.valueChanged.connect(self.change_head_length)
        self.tail_checkbox.stateChanged.connect(self.change_display_tail)
        self.id_checkbox.stateChanged.connect(self.change_display_id)
        self.graph_checkbox.stateChanged.connect(self.change_display_graph)
        self.color_by_combobox.currentTextChanged.connect(self.change_color_by)
        self.colormap_combobox.currentTextChanged.connect(self.change_colormap)

        self.layout().addRow(button_grid)
        self.layout().addRow(trans._('color by:'), self.color_by_combobox)
        self.layout().addRow(trans._('colormap:'), self.colormap_combobox)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('tail width:'), self.tail_width_slider)
        self.layout().addRow(trans._('tail length:'), self.tail_length_slider)
        self.layout().addRow(trans._('head length:'), self.head_length_slider)
        self.layout().addRow(trans._('tail:'), self.tail_checkbox)
        self.layout().addRow(trans._('show ID:'), self.id_checkbox)
        self.layout().addRow(trans._('graph:'), self.graph_checkbox)

        self._on_editable_or_visible_change()
        self._on_tail_length_change()
        self._on_tail_width_change()
        self._on_colormap_change()
        self._on_color_by_change()

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when image based layer mode changed.

        Available modes for image based layer are:
        * PAN_ZOOM
        * TRANSFORM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not PAN_ZOOM or TRANSFORM.
        """
        mode_buttons = {
            Mode.PAN_ZOOM: self.panzoom_button,
            Mode.TRANSFORM: self.transform_button,
        }

        if event.mode in mode_buttons:
            mode_buttons[event.mode].setChecked(True)
        else:
            raise ValueError(
                trans._("Mode '{mode}'not recognized", mode=event.mode)
            )

    def _on_editable_or_visible_change(self):
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )

    def _on_tail_width_change(self):
        """Receive layer model track line width change event and update slider."""
        with self.layer.events.tail_width.blocker():
            value = int(2 * self.layer.tail_width)
            self.tail_width_slider.setValue(value)

    def _on_tail_length_change(self):
        """Receive layer model track line width change event and update slider."""
        with self.layer.events.tail_length.blocker():
            value = self.layer.tail_length
            self.tail_length_slider.setValue(value)

    def _on_head_length_change(self):
        """Receive layer model track line width change event and update slider."""
        with self.layer.events.head_length.blocker():
            value = self.layer.head_length
            self.head_length_slider.setValue(value)

    def _on_properties_change(self):
        """Change the properties that can be used to color the tracks."""
        with qt_signals_blocked(self.color_by_combobox):
            self.color_by_combobox.clear()
            self.color_by_combobox.addItems(self.layer.properties_to_color_by)
        self._on_color_by_change()

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update combobox."""
        with self.layer.events.colormap.blocker():
            self.colormap_combobox.setCurrentIndex(
                self.colormap_combobox.findData(self.layer.colormap)
            )

    def _on_color_by_change(self):
        """Receive layer model color_by change event and update combobox."""
        with self.layer.events.color_by.blocker():
            color_by = self.layer.color_by

            idx = self.color_by_combobox.findText(
                color_by, Qt.MatchFlag.MatchFixedString
            )
            self.color_by_combobox.setCurrentIndex(idx)

    def change_tail_width(self, value):
        """Change track line width of shapes on the layer model.

        Parameters
        ----------
        value : float
            Line width of track tails.
        """
        self.layer.tail_width = float(value) / 2.0

    def change_tail_length(self, value):
        """Change edge line backward length of shapes on the layer model.

        Parameters
        ----------
        value : int
            Line length of track tails.
        """
        self.layer.tail_length = value

    def change_head_length(self, value):
        """Change edge line forward length of shapes on the layer model.

        Parameters
        ----------
        value : int
            Line length of track tails.
        """
        self.layer.head_length = value

    def change_display_tail(self, state):
        self.layer.display_tail = self.tail_checkbox.isChecked()

    def change_display_id(self, state):
        self.layer.display_id = self.id_checkbox.isChecked()

    def change_display_graph(self, state):
        self.layer.display_graph = self.graph_checkbox.isChecked()

    def change_color_by(self, value: str):
        self.layer.color_by = value

    def change_colormap(self, colormap: str):
        self.layer.colormap = self.colormap_combobox.currentData()
