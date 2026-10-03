from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QMessageBox,
)

from napari._qt.utils import set_widgets_enabled_with_opacity
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton
from napari.layers.base._base_constants import Mode
from napari.utils.action_manager import action_manager

if TYPE_CHECKING:
    from qtpy.QtCore import QEvent, QObject

    from napari.layers.base.base import Layer


class QtLayerButtons(QGridLayout):
    """Superclass for all the other LayerButtons classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    layer : napari.layers.Layer
        An instance of a napari layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to activate move camera mode for layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform layer.
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

        # Buttons
        self.button_group = QButtonGroup(self)
        # TODO:
        self.panzoom_button = self._radio_button(
            layer,
            'pan',
            self.MODE.PAN_ZOOM,
            False,
            self.PAN_ZOOM_ACTION_NAME,
            extra_tooltip_text='\n(or hold Space)\n(hold Shift to pan in 3D)'
            '\n(hold Alt to zoom via ROI selection)',
            checked=True,
        )
        self.transform_button = self._radio_button(
            layer,
            'transform',
            self.MODE.TRANSFORM,
            True,
            self.TRANSFORM_ACTION_NAME,
            extra_tooltip_text='\nAlt + Left mouse click over this button to reset',
        )
        self.transform_button.installEventFilter(self)
        self._on_editable_or_visible_change()

        self.addWidget(self.panzoom_button, 0, 6)
        self.addWidget(self.transform_button, 0, 7)
        self.setContentsMargins(5, 0, 0, 5)
        self.setColumnStretch(0, 1)
        self.setSpacing(4)

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
            raise ValueError(f"Mode '{event.mode}' not recognized")

    def _on_editable_or_visible_change(self):
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )
        self._set_transform_tool_state()

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

    def eventFilter(self, qobject: QObject, event: QEvent):
        """
        Event filter implementation to handle the Alt + Left mouse click interaction to
        reset the layer transform.

        For more info about Qt Event Filters you can check:
            https://doc.qt.io/qt-6/eventsandfilters.html#event-filters
        """
        if (
            qobject == self.transform_button
            and event.type() == QMouseEvent.Type.MouseButtonRelease
            and isinstance(event, QMouseEvent)
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.AltModifier
        ):
            result = QMessageBox.warning(
                self.parentWidget(),
                'Reset transform',
                'Are you sure you want to reset transforms?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self.layer._reset_affine()
                return True
        return super().eventFilter(qobject, event)


class QtMultiLayerButtons(QGridLayout):
    """Super simple shim for when multiple layers are selected.

    Effectively does nothing but shows the pan zoom button and warns
    that you're in multilayer mode.
    """

    def __init__(self, layer: Layer) -> None:
        super().__init__()
        self.button_group = QButtonGroup(self)
        btn = QtModeRadioButton(layer, 'pan', None, checked=True)
        self.button_group.addButton(btn)
        self.addWidget(btn, 0, 6)
        self.setContentsMargins(5, 0, 0, 5)
        self.setColumnStretch(0, 1)
        self.setSpacing(4)
