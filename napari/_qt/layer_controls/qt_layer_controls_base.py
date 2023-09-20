from typing import List, Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,  # TODO: Remove
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QElidingLineEdit

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
)
from napari._qt.qt_resources import QColoredSVGIcon
from napari._qt.utils import (
    set_widgets_enabled_with_opacity,
)
from napari._qt.widgets._slider_compat import QDoubleSlider  # TODO: Remove
from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
from napari.layers.base._base_constants import (
    BLENDING_TRANSLATIONS,  # TODO: Remove
    Blending,  # TODO: Remove
    Mode,
)
from napari.layers.base.base import Layer
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.events import Event, disconnect_events
from napari.utils.misc import StringEnum
from napari.utils.translations import trans

# TODO: Remove
# opaque and minimum blending do not support changing alpha (opacity)
NO_OPACITY_BLENDING_MODES = {str(Blending.MINIMUM), str(Blending.OPAQUE)}


class LayerFormLayout(QFormLayout):
    """Reusable form layout for subwidgets in each QtLayerControls class"""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(5)  # Spacing between rows
        self.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # Needed since default aligment depends on OS (win/linux left, macos right)
        self.setLabelAlignment(Qt.AlignmentFlag.AlignRight)


class QtCollapsibleLayerControlsSection(QCollapsible):
    """
    Customization of the QCollapsible class to set default icons and style.
    Uses a `LayerFormLayout` internally to organize added widgets to create
    layer controls collapsible sections. See `addRowToSection`
    """

    def __init__(self, title: str = "", parent: QWidget = None) -> None:
        super().__init__(title=title, parent=parent)
        # Use `clicked` instead of `toggled` to prevent `QPropertyAnimation` leak
        self._toggle_btn.toggled.disconnect()
        self._toggle_btn.clicked.connect(self._toggle)
        # Set themed icons
        # TODO: Is there a better way to handle a theme change to set icons?
        get_settings().appearance.events.theme.connect(self.setThemedIcons)
        self.setThemedIcons()

        # Setup internal layout
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.content().layout().setContentsMargins(0, 0, 0, 0)
        self.setProperty('emphasized', True)
        form_widget = QWidget()
        form_widget.setProperty('emphasized', True)
        self._internal_layout = LayerFormLayout()
        self._internal_layout.setContentsMargins(36, 10, 18, 13)
        form_widget.setLayout(self._internal_layout)
        self.addWidget(form_widget)
        self.expand(animate=False)

    # ---- Overridden methods
    def expand(self, animate: bool = True) -> None:
        super().expand(animate=animate)
        self._toggle_btn.setToolTip(
            trans._("Collapse {title} layer controls", title=self._text)
        )
        self._content.show()

    def collapse(self, animate: bool = True) -> None:
        super().collapse(animate=animate)
        self._toggle_btn.setToolTip(
            trans._("Expand {title} layer controls", title=self._text)
        )
        self._content.hide()

    # ---- New methods to follow napari theme and enable easy widget addition
    def setThemedIcons(self, theme_event: Event = None) -> None:
        """
        Set the correct icons for the widget toggle button following the given
        theme event value or the current theme in the settings.

        Parameters
        ----------
        theme_event : Event, optional
            Event with the new theme value to use. The default is None.
        """
        if theme_event:
            theme = theme_event.value
        else:
            theme = get_settings().appearance.theme
        coll_icon = QColoredSVGIcon.from_resources('right_arrow').colored(
            theme=theme
        )
        exp_icon = QColoredSVGIcon.from_resources('down_arrow').colored(
            theme=theme
        )
        self.setCollapsedIcon(icon=coll_icon)
        self.setExpandedIcon(icon=exp_icon)

    def addRowToSection(self, *args) -> None:
        """
        Add a new row to the bottom of the internal `LayerFormLayout` with the
        given arguments.

        Parameters
        ----------
        *args :
            Arguments that a `QFormLayout` expects. For more information you
            can check https://doc.qt.io/qt-5/qformlayout.html#addRow
        """
        self._internal_layout.addRow(*args)


class QtLayerControls(QFrame):  # TODO: Remove
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    layer : napari.layers.Layer
        An instance of a napari layer.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    opacityLabel : qtpy.QtWidgets.QLabel
        Label for the opacity slider widget.
    """

    def __init__(self, layer: Layer) -> None:
        super().__init__()

        self._ndisplay: int = 2

        self.layer = layer
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.opacity.connect(self._on_opacity_change)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.setLayout(LayerFormLayout(self))

        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=self)
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
        to 2D or 3D visualization only.
        """

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


class NewQtLayerControls(
    QtLayerControls
):  # TODO: Inherit from `QFrame` directly
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.
    mode_options: napari.utils.misc.StringEnum
        Enum definition with the layer modes. Default enum counts with `PAN_ZOOM`
        and `TRANSFORM` modes values.

    Raises
    ------
    ValueError
        Raise error if layer mode is not recognized.
    """

    # Enable setting specific Mode enum type but also define as
    # default one the base one with only PAN_ZOOM and TRANSFORM values
    # to create base buttons
    def __init__(self, layer: Layer, mode_options: StringEnum = Mode) -> None:
        # TODO: Restore super call without args
        # super().__init__()
        super(QtLayerControls, self).__init__()
        # Base attributes
        self._edit_buttons: list = []
        self._mode_buttons: dict = {}
        self._ndisplay: int = 2
        self._widget_controls: list = []
        self._layer = layer
        self._mode_options = mode_options

        # Layer base events connection
        self._layer.events.editable.connect(
            self._on_editable_or_visible_change
        )
        self._layer.events.name.connect(self._on_name_change)
        self._layer.events.mode.connect(self._on_mode_change)
        self._layer.events.visible.connect(self._on_editable_or_visible_change)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        # Setup layer name section
        name_layout = QHBoxLayout()
        # Needed to put icon and text side by side with the same spacing in all platforms
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(0)

        icon_label = QLabel()
        icon_label.setProperty('layer_type_icon_label', True)
        icon_label.setObjectName(f'{self._layer._basename()}')

        self._name_label = QElidingLineEdit(self._layer.name)
        self._name_label.setToolTip(self._layer.name)
        self._name_label.setObjectName('layer_name')
        self._name_label.textChanged.connect(self._on_widget_name_change)
        self._name_label.editingFinished.connect(self.setFocus)

        name_layout.addWidget(icon_label)
        name_layout.addWidget(self._name_label)
        name_layout.addStretch(1)

        # Setup buttons section
        self._buttons_grid = QGridLayout()
        self._buttons_grid.setContentsMargins(0, 0, 0, 0)
        # Need to set spacing to have same spacing over all platforms
        self._buttons_grid.setSpacing(10)  # +-6 win/linux def; +-15 macos def
        self._buttons_grid.setHorizontalSpacing(9)
        # Need to set strech for a last column to prevent the spacing between
        # buttons to change when the layer control width changes
        self._buttons_grid.setColumnStretch(7, 1)
        self._button_group = QButtonGroup(self)

        # Setup layer controls sections
        self._annotation_controls_section = QtCollapsibleLayerControlsSection(
            trans._("annotation")
        )
        self._display_controls_section = QtCollapsibleLayerControlsSection(
            trans._("display")
        )
        controls_scrollarea = QScrollArea()
        controls_scrollarea.setWidgetResizable(True)
        controls_scrollarea.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scrollarea.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 5, 0)
        controls_layout.setSpacing(4)
        controls_layout.addWidget(self._annotation_controls_section)
        controls_layout.addWidget(self._display_controls_section)
        controls_layout.addStretch(1)
        controls_widget.setLayout(controls_layout)
        controls_scrollarea.setWidget(controls_widget)

        self._annotation_controls_section.hide()
        self._display_controls_section.hide()

        # Setup base layout
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(8, 0, 0, 0)
        # Spacing between sections (layer name, buttons and controls)
        self.layout().setSpacing(10)
        self.layout().addLayout(name_layout)
        self.layout().addLayout(self._buttons_grid)
        self.layout().addWidget(controls_scrollarea)

    def __getattr__(self, attr: str):
        """
        Redefinition of __getattr__ to enable access to widget controls.
        """
        for widget_control in self._widget_controls:
            widget_attr = getattr(widget_control, attr, None)
            if widget_attr:
                return widget_attr
        return super().__getattr__(attr)

    def _add_radio_button_mode(
        self,
        btn_name: str,
        mode: StringEnum,
        action_name: str,
        row: int,
        column: int,
        extra_tooltip_text: str = '',
        edit_button: bool = True,
        **kwargs,
    ) -> QtModeRadioButton:
        """
        Convenience method to create a QtModeRadioButton, add it to the buttons layout
        and bind it to an action at the same time.

        Parameters
        ----------
        btn_name : str
            name fo the button
        mode : Enum
            Value Associated to current button
        action_name : str
            Action triggered when button pressed
        row : int
            Row to position the button in the buttons layout
        column : int
            Column to position the button in the buttons layout
        extra_tooltip_text : str, optional
            Text you want added after the automatic tooltip set by the
            action manager
        edit_button : bool, optional
            If the button is related with edition actions and should handle
            a disable state when the layer is not visible.
        **kwargs:
            Passed to napari._qt.widgets.qt_mode_buttons.QtModeRadioButton

        Returns
        -------
        button: napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
            button bound (or that will be bound to) to action `action_name`

        Notes
        -----
        When shortcuts are modifed/added/removed via the action manager, the
        tooltip will be updated to reflect the new shortcut.
        """
        action_name = f'napari:{action_name}'
        btn = QtModeRadioButton(self._layer, btn_name, mode, **kwargs)
        action_manager.bind_button(
            action_name,
            btn,
            extra_tooltip_text=extra_tooltip_text,
        )
        self._mode_buttons[mode] = btn
        if edit_button:
            self._edit_buttons.append(btn)
        self._button_group.addButton(btn)
        self._buttons_grid.addWidget(btn, row, column)

        return btn

    def _add_push_button_action(
        self,
        btn_name: str,
        row: int,
        column: int,
        action_name: Optional[str] = None,
        slot: Optional[callable] = None,
        tooltip: str = '',
        edit_button: bool = True,
    ) -> QtModePushButton:
        """
        Convenience method to create a QtModePushButton, add it to the buttons layout
        and bind it to an action at the same time if necessary.

        Parameters
        ----------
        btn_name : str
            Name for the button.  This is mostly used to identify the button
            in stylesheets (e.g. to add a custom icon).
        row : int
            Row to position the button in the buttons layout
        column : int
            Column to position the button in the buttons layout
        action_name : str, optional
            Action triggered when button pressed.
        slot : callable, optional
            The function to call when this button is clicked.
        tooltip : str, optional
            A tooltip to display when hovering the mouse on this button.
        edit_button : bool, optional
            If the button is related with edition actions and should handle
            a disable state when the layer is not visible.

        Returns
        -------
        button: napari._qt.widgets.qt_mode_buttons.QtModePushButton
            button bound (or that will be bound to) to action `action_name`

        """
        btn = QtModePushButton(
            self._layer,
            btn_name,
            slot=slot,
            tooltip=tooltip,
        )
        if action_name:
            action_name = f'napari:{action_name}'
            action_manager.bind_button(action_name, btn)
        if edit_button:
            self._edit_buttons.append(btn)
        self._buttons_grid.addWidget(btn, row, column)

        return btn

    def _add_widget_controls(
        self,
        section_att: str,
        wrapper: QtWidgetControlsBase,
        controls: Optional[List[Tuple[QLabel, QWidget]]] = None,
        add_wrapper: bool = True,
    ) -> None:
        """
        Add widget controls to the given collapsible controls section.

        If the section is not visible when adding a control visibility is changed.

        Parameters
        ----------
        section_att : str
            Attribute of the section where the controls should be added.
            It should be either `_annotation_controls_section` or `_display_controls_section`
        wrapper : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWidgetControlsBase
            An instance of a `QtWidgetControlsBase` subclass that setups
            widgets for a layer attribute.
        controls : List[Tuple[QLabel, QWidget]]
            A list of widget controls tuples. Each tuple has the label for the
            control and the respective control widget to show.
        add_wrapper : bool
            True if a reference to the wrapper class should be kept.
            False otherwise.
        """
        if controls is None:
            controls = []
        section = getattr(self, section_att)

        if add_wrapper:
            self._widget_controls.append(wrapper)

        if len(controls) == 0:
            controls = wrapper.get_widget_controls()

        for label_text, control_widget in controls:
            section.addRowToSection(label_text, control_widget)
        if not section.isVisible():
            section.show()

    def _on_mode_change(self, event: Event) -> None:
        """Update checked button mode when layer mode changes.

        Available default modes for a layer are:
            * PAN_ZOOM
            * TRANSFORM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not recognized.
        """
        if event.mode in self._mode_buttons:
            self._mode_buttons[event.mode].setChecked(True)
        elif event.mode != self._mode_options.TRANSFORM:
            raise ValueError(
                trans._("Mode '{mode}' not recognized", mode=event.mode)
            )

    def _on_editable_or_visible_change(self) -> None:
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._edit_buttons,
            self._layer.editable and self._layer.visible,
        )

    def _on_widget_name_change(self, text) -> None:
        """
        Receive widget name change signal and update layer name.

        Also, update widget tooltip with new full text (without ellipsis).

        Parameters
        ----------
        text : str
            Updated text. Not used here to prevent setting text with ellipsis.

        """
        with self._layer.events.blocker(self._on_name_change):
            new_name = self._name_label.text()
            self._layer.name = new_name
            self._name_label.setToolTip(new_name)

    def _on_name_change(self) -> None:
        """Receive layer model name change event and update name label and tooltip."""
        self._name_label.setText(self._layer.name)
        self._name_label.setToolTip(self._layer.name)

    def _on_ndisplay_changed(self) -> None:
        """Respond to a change to the number of dimensions displayed in the viewer.

        This is needed because some layer controls may have options that are specific
        to 2D or 3D visualization only.
        """

    @property
    def ndisplay(self) -> int:
        """The number of dimensions displayed in the canvas."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        self._ndisplay = ndisplay
        self._on_ndisplay_changed()

    def add_annotation_widget_controls(
        self,
        wrapper: QtWidgetControlsBase,
        controls: Optional[List[Tuple[QLabel, QWidget]]] = None,
        add_wrapper: bool = True,
    ) -> None:
        """
        Add widget controls to the collapsible annotation controls section.

        Parameters
        ----------
        wrapper : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWidgetControlsBase
            An instance of a `QtWidgetControlsBase` subclass that setups
            widgets for a layer attribute.
        controls : List[Tuple[QLabel, QWidget]]
            A list of widget controls tuples. Each tuple has the label for the
            control and the respective control widget to show.
        add_wrapper : bool
            True if a reference to the wrapper class should be kept.
            False otherwise.
        """
        if controls is None:
            controls = []
        self._add_widget_controls(
            "_annotation_controls_section",
            wrapper,
            controls=controls,
            add_wrapper=add_wrapper,
        )

    def add_display_widget_controls(
        self,
        wrapper: QtWidgetControlsBase,
        controls: Optional[List[Tuple[QLabel, QWidget]]] = None,
        add_wrapper: bool = True,
    ) -> None:
        """
        Add widget controls to the collapsible display controls section.

        Parameters
        ----------
        wrapper : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWidgetControlsBase
            An instance of a `QtWidgetControlsBase` subclass that setups
            widgets for a layer attribute.
        controls : List[Tuple[QLabel, QWidget]]
            A list of widget controls tuples. Each tuple has the label for the
            control and the respective control widget to show.
        add_wrapper : bool
            True if a reference to the wrapper class should be kept.
            False otherwise.
        """
        if controls is None:
            controls = []
        self._add_widget_controls(
            "_display_controls_section",
            wrapper,
            controls=controls,
            add_wrapper=add_wrapper,
        )

    def on_theme_changed(self, event: Event) -> None:
        """
        Handle theme changes.

        Needed to update the icon in the collapsible widget controls sections.

        Parameters
        ----------
        event : napari.utils.events.Event
            Theme event.
        """
        self._annotation_controls_section.setThemedIcons(event)
        self._display_controls_section.setThemedIcons(event)

    def deleteLater(self) -> None:
        disconnect_events(self._layer.events, self)
        for widget_control in self._widget_controls:
            widget_control.disconnect_widget_controls()
        super(
            QtLayerControls, self
        ).deleteLater()  # TODO: Call super without args

    def close(self) -> bool:
        """Disconnect events when widget is closing."""
        for widget_control in self._widget_controls:
            widget_control.disconnect_widget_controls()
        for child in self.children():
            close_method = getattr(child, 'close', None)
            if close_method is not None:
                close_method()
        return super(
            QtLayerControls, self
        ).close()  # TODO: Call super without args
