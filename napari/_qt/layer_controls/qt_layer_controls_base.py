from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
)
from superqt import QLabeledDoubleSlider

from napari._qt.utils import set_widgets_enabled_with_opacity
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton
from napari.layers.base._base_constants import (
    BLENDING_TRANSLATIONS,
    Blending,
    Mode,
)
from napari.layers.base.base import Layer
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.translations import trans

# opaque and minimum blending do not support changing alpha (opacity)
NO_OPACITY_BLENDING_MODES = {str(Blending.MINIMUM), str(Blending.OPAQUE)}


class LayerFormLayout(QFormLayout):
    """Reusable form layout for subwidgets in each QtLayerControls class"""

    def __init__(self, QWidget=None) -> None:
        super().__init__(QWidget)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)
        self.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)


class QtLayerControls(QFrame):
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
    blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    layer : napari.layers.Layer
        An instance of a napari layer.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    opacityLabel : qtpy.QtWidgets.QLabel
        Label for the opacity slider widget.
    """

    MODE = Mode
    PAN_ZOOM_ACTION_NAME = ''
    TRANSFORM_ACTION_NAME = ''

    def __init__(self, layer: Layer) -> None:
        super().__init__()

        self._ndisplay: int = 2
        self._EDIT_BUTTONS: tuple = ()
        self._MODE_BUTTONS: dict = {}

        self.layer = layer
        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.editable.connect(self._on_editable_or_visible_change)
        self.layer.events.visible.connect(self._on_editable_or_visible_change)
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.opacity.connect(self._on_opacity_change)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.setLayout(LayerFormLayout(self))

        # Buttons
        self.button_group = QButtonGroup(self)
        self.panzoom_button = self._radio_button(
            layer,
            'pan',
            self.MODE.PAN_ZOOM,
            False,
            self.PAN_ZOOM_ACTION_NAME,
            extra_tooltip_text=trans._('(or hold Space)'),
            checked=True,
        )
        self.transform_button = self._radio_button(
            layer,
            'transform',
            self.MODE.TRANSFORM,
            True,
            self.TRANSFORM_ACTION_NAME,
            extra_tooltip_text=trans._(
                '\nAlt + Left mouse click over this button to reset'
            ),
        )
        self.transform_button.installEventFilter(self)
        self._on_editable_or_visible_change()

        self.button_grid = QGridLayout()
        self.button_grid.addWidget(self.panzoom_button, 0, 6)
        self.button_grid.addWidget(self.transform_button, 0, 7)
        self.button_grid.setContentsMargins(5, 0, 0, 5)
        self.button_grid.setColumnStretch(0, 1)
        self.button_grid.setSpacing(4)

        # Control widgets
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(1)
        sld.setSingleStep(0.01)
        sld.valueChanged.connect(self.changeOpacity)
        self.opacitySlider = sld
        self.opacityLabel = QLabel(trans._('opacity:'))

        self._on_opacity_change()

        blend_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(BLENDING_TRANSLATIONS.items()):
            data = data.value
            blend_comboBox.addItem(text, data)
            if data == self.layer.blending:
                blend_comboBox.setCurrentIndex(index)

        blend_comboBox.currentTextChanged.connect(self.changeBlending)
        self.blendComboBox = blend_comboBox
        # opaque and minimum blending do not support changing alpha
        self.opacitySlider.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacityLabel.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        if self.__class__ == QtLayerControls:
            # This base class is only instantiated in tests. When it's not a
            # concrete subclass, we need to parent the button_grid to the
            # layout so that qtbot will correctly clean up all instantiated
            # widgets.
            self.layout().addRow(self.button_grid)

    def changeOpacity(self, value):
        """Change opacity value on the layer model.

        Parameters
        ----------
        value : float
            Opacity value for shapes.
            Input range 0 - 100 (transparent to fully opaque).
        """
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.opacity = value

    def changeBlending(self, text):
        """Change blending mode on the layer model.

        Parameters
        ----------
        text : str
            Name of blending mode, eg: 'translucent', 'additive', 'opaque'.
        """
        self.layer.blending = self.blendComboBox.currentData()
        # opaque and minimum blending do not support changing alpha
        self.opacitySlider.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )
        self.opacityLabel.setEnabled(
            self.layer.blending not in NO_OPACITY_BLENDING_MODES
        )

        blending_tooltip = ''
        if self.layer.blending == str(Blending.MINIMUM):
            blending_tooltip = trans._(
                '`minimum` blending mode works best with inverted colormaps with a white background.',
            )
        self.blendComboBox.setToolTip(blending_tooltip)
        self.layer.help = blending_tooltip

    def _radio_button(
        self,
        layer,
        btn_name,
        mode,
        edit_button,
        action_name,
        extra_tooltip_text='',
        **kwargs,
    ):
        """
        Convenience local function to create a RadioButton and bind it to
        an action at the same time.

        Parameters
        ----------
        layer : napari.layers.Layer
            The layer instance that this button controls.n
        btn_name : str
            name fo the button
        mode : Enum
            Value Associated to current button
        edit_button: bool
            True if the button corresponds to edition operations. False otherwise.
        action_name : str
            Action triggered when button pressed
        extra_tooltip_text : str
            Text you want added after the automatic tooltip set by the
            action manager
        **kwargs:
            Passed to napari._qt.widgets.qt_mode_button.QtModeRadioButton

        Returns
        -------
        button: napari._qt.widgets.qt_mode_button.QtModeRadioButton
            button bound (or that will be bound to) to action `action_name`

        Notes
        -----
        When shortcuts are modifed/added/removed via the action manager, the
        tooltip will be updated to reflect the new shortcut.
        """
        action_name = f'napari:{action_name}'
        btn = QtModeRadioButton(layer, btn_name, mode, **kwargs)
        action_manager.bind_button(
            action_name,
            btn,
            extra_tooltip_text=extra_tooltip_text,
        )
        self._MODE_BUTTONS[mode] = btn
        self.button_group.addButton(btn)
        if edit_button:
            self._EDIT_BUTTONS += (btn,)
        return btn

    def _on_mode_change(self, event):
        """
        Update ticks in checkbox widgets when image based layer mode changed.

        Available modes for base layer are:
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
        if event.mode in self._MODE_BUTTONS:
            self._MODE_BUTTONS[event.mode].setChecked(True)
        else:
            raise ValueError(
                trans._("Mode '{mode}' not recognized", mode=event.mode)
            )

    def _on_editable_or_visible_change(self):
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )
        self._set_transform_tool_state()

    def _on_opacity_change(self):
        """Receive layer model opacity change event and update opacity slider."""
        with self.layer.events.opacity.blocker():
            self.opacitySlider.setValue(self.layer.opacity)

    def _on_blending_change(self):
        """Receive layer model blending mode change event and update slider."""
        with self.layer.events.blending.blocker():
            self.blendComboBox.setCurrentIndex(
                self.blendComboBox.findData(self.layer.blending)
            )

    @property
    def ndisplay(self) -> int:
        """The number of dimensions displayed in the canvas."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        self._ndisplay = ndisplay
        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self) -> None:
        """Respond to a change to the number of dimensions displayed in the viewer.

        This is needed because some layer controls may have options that are specific
        to 2D or 3D visualization only like the transform mode button.
        """
        self._set_transform_tool_state()

    def _set_transform_tool_state(self):
        """
        Enable/disable transform button taking into account:
            * Layer visibility.
            * Layer editability.
            * Number of dimensions being displayed.
        """
        set_widgets_enabled_with_opacity(
            self,
            [self.transform_button],
            self.layer.editable and self.layer.visible and self.ndisplay == 2,
        )

    def eventFilter(self, qobject, event):
        """
        Event filter implementation to handle the Alt + Left mouse click interaction to
        reset the layer transform.

        For more info about Qt Event Filters you can check:
            https://doc.qt.io/qt-6/eventsandfilters.html#event-filters
        """
        if (
            qobject == self.transform_button
            and event.type() == QMouseEvent.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.AltModifier
        ):
            result = QMessageBox.warning(
                self,
                trans._('Reset transform'),
                trans._('Are you sure you want to reset transforms?'),
                QMessageBox.Yes | QMessageBox.No,
            )
            if result == QMessageBox.Yes:
                self.layer._reset_affine()
                return True
        return super().eventFilter(qobject, event)

    def deleteLater(self):
        disconnect_events(self.layer.events, self)
        super().deleteLater()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        for child in self.children():
            close_method = getattr(child, 'close', None)
            if close_method is not None:
                close_method()
        return super().close()
