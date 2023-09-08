from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
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

from napari._qt.layer_controls.widgets import QtOpacityBlendingControls
from napari._qt.qt_resources import QColoredSVGIcon
from napari._qt.utils import (
    set_widgets_enabled_with_opacity,
)
from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
from napari.layers.base._base_constants import Mode
from napari.layers.base.base import Layer
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.events import Event, disconnect_events
from napari.utils.misc import StringEnum
from napari.utils.translations import trans


class LayerFormLayout(QFormLayout):
    """Reusable form layout for subwidgets in each QtLayerControls class"""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)
        self.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # Needed since default aligment depends on OS
        # self.setLabelAlignment(Qt.AlignmentFlag.AlignRight)


class QtCollapsibleLayerControlsSection(QCollapsible):
    """
    Customization of the QCollapsible class to set default icons and style.
    Uses a `LayerFormLayout` internally to organize added widgets to create
    layer controls collapsible sections. See `addRowToSection`
    """

    def __init__(self, title: str = "", parent: QWidget = None) -> None:
        super().__init__(
            title=title,
            parent=parent,
        )
        # Set themed icons
        # TODO: Is there a better way to handle a theme change to set icons?
        self._setIconsByTheme()
        get_settings().appearance.events.theme.connect(self._setIconsByTheme)

        # Setup internal layout
        self.content().layout().setContentsMargins(0, 0, 0, 0)
        self.setProperty('emphasized', True)
        form_widget = QWidget()
        form_widget.setProperty('emphasized', True)
        self._internal_layout = LayerFormLayout()
        form_widget.setLayout(self._internal_layout)
        self.addWidget(form_widget)

        self.expand()

    # ---- Overridden methods
    def expand(self, animate: bool = True) -> None:
        super().expand(animate=animate)
        self._toggle_btn.setToolTip(
            trans._("Collapse {title} controls", title=self._text)
        )

    def collapse(self, animate: bool = True) -> None:
        super().collapse(animate=animate)
        self._toggle_btn.setToolTip(
            trans._("Expand {title} controls", title=self._text)
        )

    # ---- New methods to follow napari theme and enable easy widget addition
    def _setIconsByTheme(self, theme_event: Event = None) -> None:
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
        self._internal_layout.addRow(*args)


class QtLayerControls(QFrame):
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.
    mode_options: napari.utils.misc.StringEnum
        Enum definition with the layer modes. Default enum counts with PAN_ZOOM
        and TRANSFORM modes values.
    """

    # Enable setting expecific Mode enum type but also define as
    # default one the base one with only PAN_ZOOM and TRANSFORM values
    # to create base buttons
    def __init__(self, layer: Layer, mode_options: StringEnum = Mode) -> None:
        super().__init__()
        # Base attributes
        self._edit_buttons: list = []
        self._mode_buttons: dict = {}
        self._ndisplay: int = 2
        self._widget_controls: dict = {}
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

        # Setup buttons section
        self._buttons_grid = QGridLayout()
        self._buttons_grid.setContentsMargins(0, 0, 0, 0)
        # Need to set spacing to have same spacing over all platforms
        self._buttons_grid.setSpacing(10)  # +-6 win/linux def; +-15 macos def
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
        # TODO: Should the scrollbar be always visible?
        controls_scrollarea.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 5, 0)
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
        self.layout().addLayout(name_layout)
        self.layout().addLayout(self._buttons_grid)
        self.layout().addWidget(controls_scrollarea)

        # Setup base widget controls
        # TODO: Should be done when instanciating layer controls class
        # at the layer controls container via some sort of mapping between
        # layer attributes and QObject classes with QWidgets-Layer atts
        # connection logic
        self._widget_controls[
            "opacity_blending_controls"
        ] = QtOpacityBlendingControls(self, layer)

    def _radio_button_mode(
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
        Convenience function to create a RadioButton and bind it to
        an action at the same time.

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

    def _push_button_action(
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
        Convenience function to create a PushButton and bind it to
        an action at the same time.

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
        button: QtModeRadioButton
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

    def _on_mode_change(self, event: Event) -> None:
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
        self, controls: list[tuple[QLabel, QWidget]]
    ) -> None:
        """
        Add controls to the collapsible annotation controls section.

        Parameters
        ----------
        controls : list[tuple[QLabel, QWidget]]
            A list of widget controls tuples. Each tuple has the label for the
            control and the respective control widget to show.
        """
        for label_text, control_widget in controls:
            self._annotation_controls_section.addRowToSection(
                label_text, control_widget
            )
        if not self._annotation_controls_section.isVisible():
            self._annotation_controls_section.show()

    def add_display_widget_controls(
        self, controls: list[tuple[QLabel, QWidget]]
    ) -> None:
        """
        Add widget controls to the collapsible display controls section.

        Parameters
        ----------
        controls : list[tuple[QLabel, QWidget]]
            A list of widget controls tuples. Each tuple has the label for the
            control and the respective control widget to show.
        """
        for label_text, control_widget in controls:
            self._display_controls_section.addRowToSection(
                label_text, control_widget
            )
        if not self._display_controls_section.isVisible():
            self._display_controls_section.show()

    def deleteLater(self) -> None:
        disconnect_events(self._layer.events, self)
        super().deleteLater()

    def close(self) -> bool:
        """Disconnect events when widget is closing."""
        for widget_control in self._widget_controls.values():
            widget_control.disconnect_widget_controls()
        for child in self.children():
            close_method = getattr(child, 'close', None)
            if close_method is not None:
                close_method()
        return super().close()
