import warnings
from enum import Enum, EnumMeta
from functools import partial, wraps
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)
from superqt import QEnumComboBox, QLabeledDoubleSlider

from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.widgets.qt_dims_sorter import QtDimsSorter
from napari._qt.widgets.qt_spinbox import QtSpinBox
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.utils.action_manager import action_manager
from napari.utils.camera_orientations import (
    DepthAxisOrientation,
    DepthAxisOrientationStr,
    HorizontalAxisOrientation,
    HorizontalAxisOrientationStr,
    VerticalAxisOrientation,
    VerticalAxisOrientationStr,
)
from napari.utils.misc import in_ipython, in_jupyter, in_python_repl
from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from napari.viewer import ViewerModel


def add_new_points(viewer):
    viewer.add_points(
        ndim=max(viewer.dims.ndim, 2),
        scale=viewer.layers.extent.step,
    )


def add_new_shapes(viewer):
    viewer.add_shapes(
        ndim=max(viewer.dims.ndim, 2),
        scale=viewer.layers.extent.step,
    )


class QtLayerButtons(QFrame):
    """Button controls for napari layers.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    deleteButton : QtDeleteButton
        Button to delete selected layers.
    newLabelsButton : QtViewerPushButton
        Button to add new Label layer.
    newPointsButton : QtViewerPushButton
        Button to add new Points layer.
    newShapesButton : QtViewerPushButton
        Button to add new Shapes layer.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer: 'ViewerModel') -> None:
        super().__init__()

        self.viewer = viewer

        self.deleteButton = QtViewerPushButton(
            'delete_button', action='napari:delete_selected_layers'
        )

        self.newPointsButton = QtViewerPushButton(
            'new_points',
            trans._('New points layer'),
            partial(add_new_points, self.viewer),
        )

        self.newShapesButton = QtViewerPushButton(
            'new_shapes',
            trans._('New shapes layer'),
            partial(add_new_shapes, self.viewer),
        )
        self.newLabelsButton = QtViewerPushButton(
            'new_labels',
            trans._('New labels layer'),
            self.viewer._new_labels,
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.newPointsButton)
        layout.addWidget(self.newShapesButton)
        layout.addWidget(self.newLabelsButton)
        layout.addStretch(0)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)


def labeled_double_slider(
    *,
    parent: QtPopup,
    value: float,
    value_range: tuple[float, float],
    decimals: int = 0,
    callback: 'Callable',
) -> QLabeledDoubleSlider:
    """Create a labeled double slider widget."""
    slider = QLabeledDoubleSlider(parent)
    slider.setRange(*value_range)
    slider.setValue(value)
    slider.setDecimals(decimals)
    slider.valueChanged.connect(callback)
    return slider


def enum_combobox(
    *,
    parent: QtPopup,
    enum_class: EnumMeta,
    current_enum: Enum,
    callback: 'Callable[[],Any] | Callable[[Enum],Any]',
) -> QEnumComboBox:
    """Create an enum combobox widget."""
    combo = QEnumComboBox(parent, enum_class=enum_class)
    combo.setCurrentEnum(current_enum)
    combo.currentEnumChanged.connect(callback)
    return combo


def help_tooltip(
    *,
    parent: QtPopup,
    text: str,
    object_name: str = 'help_label',
) -> QtToolTipLabel:
    """Create a help tooltip widget."""
    help_symbol = QtToolTipLabel(parent)
    help_symbol.setObjectName(object_name)
    help_symbol.setToolTip(text)
    return help_symbol


class QtViewerButtons(QFrame):
    """Button controls for the napari viewer.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    consoleButton : QtViewerPushButton
        Button to open iPython console within napari.
    rollDimsButton : QtViewerPushButton
        Button to roll orientation of spatial dimensions in the napari viewer.
    transposeDimsButton : QtViewerPushButton
        Button to transpose dimensions in the napari viewer.
    resetViewButton : QtViewerPushButton
        Button resetting the view of the rendered scene.
    gridViewButton : QtViewerPushButton
        Button to toggle grid view mode of layers on and off.
    ndisplayButton : QtViewerPushButton
        Button to toggle number of displayed dimensions.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer: 'ViewerModel') -> None:
        super().__init__()

        self.viewer = viewer

        self.consoleButton = QtViewerPushButton(
            'console', action='napari:toggle_console_visibility'
        )
        self.consoleButton.setProperty('expanded', False)
        if in_ipython() or in_jupyter() or in_python_repl():
            self.consoleButton.setEnabled(False)

        rdb = QtViewerPushButton('roll', action='napari:roll_axes')
        self.rollDimsButton = rdb
        rdb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        rdb.customContextMenuRequested.connect(self._open_roll_popup)

        self.transposeDimsButton = QtViewerPushButton(
            'transpose',
            action='napari:transpose_axes',
            extra_tooltip_text=trans._(
                '\nAlt/option-click to rotate visible axes'
            ),
        )
        self.transposeDimsButton.installEventFilter(self)

        self.resetViewButton = QtViewerPushButton(
            'home', action='napari:reset_view'
        )
        gvb = QtViewerPushButton(
            'grid_view_button', action='napari:toggle_grid'
        )
        self.gridViewButton = gvb
        gvb.setCheckable(True)
        gvb.setChecked(viewer.grid.enabled)
        gvb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        gvb.customContextMenuRequested.connect(self._open_grid_popup)

        @self.viewer.grid.events.enabled.connect
        def _set_grid_mode_checkstate(event):
            gvb.setChecked(event.value)

        ndb = QtViewerPushButton(
            'ndisplay_button', action='napari:toggle_ndisplay'
        )
        self.ndisplayButton = ndb
        ndb.setCheckable(True)
        ndb.setChecked(self.viewer.dims.ndisplay == 3)
        ndb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        ndb.customContextMenuRequested.connect(self.open_ndisplay_camera_popup)

        @self.viewer.dims.events.ndisplay.connect
        def _set_ndisplay_mode_checkstate(event):
            ndb.setChecked(event.value == 3)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.consoleButton)
        layout.addWidget(self.ndisplayButton)
        layout.addWidget(self.rollDimsButton)
        layout.addWidget(self.transposeDimsButton)
        layout.addWidget(self.gridViewButton)
        layout.addWidget(self.resetViewButton)
        layout.addStretch(0)
        self.setLayout(layout)

    def eventFilter(self, qobject, event):
        """Have Alt/Option key rotate layers with the transpose button."""
        modifiers = QApplication.keyboardModifiers()
        if (
            modifiers == Qt.AltModifier
            and qobject == self.transposeDimsButton
            and event.type() == QEvent.MouseButtonPress
        ):
            action_manager.trigger('napari:rotate_layers')
            return True
        return False

    def _position_popup_inside_viewer(
        self, popup: QtPopup, button: QPushButton
    ) -> None:
        """Position the popup so that it remains with the viewer.

        Use the location of the button as a reference and place the popup
        above it. move_to will adjust the position if the popup is too close
        to the edge of the screen.
        """
        button_rect = button.rect()
        button_pos = button.mapToGlobal(button_rect.topLeft())
        popup.move_to(
            (
                button_pos.x(),
                button_pos.y() - popup.sizeHint().height() - 5,
                popup.sizeHint().width(),
                popup.sizeHint().height(),
            )
        )
        popup.show()

    def _add_3d_camera_controls(
        self,
        popup: QtPopup,
        grid_layout: QGridLayout,
    ) -> None:
        """Add 3D camera controls to the popup."""
        self.perspective = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.perspective,
            value_range=(0, 90),
            callback=self._update_perspective,
        )

        perspective_help_symbol = help_tooltip(
            parent=popup,
            text='Controls perspective projection strength. 0 is orthographic, larger values increase perspective effect.',
        )

        self.rx = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.angles[0],
            value_range=(-180, 180),
            callback=partial(self._update_camera_angles, 0),
        )

        self.ry = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.angles[1],
            value_range=(-89, 89),
            callback=partial(self._update_camera_angles, 1),
        )

        self.rz = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.angles[2],
            value_range=(-180, 180),
            callback=partial(self._update_camera_angles, 2),
        )

        angle_help_symbol = help_tooltip(
            parent=popup,
            text='Controls the rotation angles around each axis in degrees.',
        )

        grid_layout.addWidget(QLabel(trans._('Perspective:')), 2, 0)
        grid_layout.addWidget(self.perspective, 2, 1)
        grid_layout.addWidget(perspective_help_symbol, 2, 2)

        grid_layout.addWidget(QLabel(trans._('Angles    X:')), 3, 0)
        grid_layout.addWidget(self.rx, 3, 1)
        grid_layout.addWidget(angle_help_symbol, 3, 2)

        grid_layout.addWidget(QLabel(trans._('             Y:')), 4, 0)
        grid_layout.addWidget(self.ry, 4, 1)

        grid_layout.addWidget(QLabel(trans._('             Z:')), 5, 0)
        grid_layout.addWidget(self.rz, 5, 1)

    def _add_shared_camera_controls(
        self,
        popup: QtPopup,
        grid_layout: QGridLayout,
    ) -> None:
        """Add shared camera controls to the popup."""
        self.zoom = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.zoom,
            value_range=(0.01, 100),
            decimals=2,
            callback=self._update_zoom,
        )

        zoom_help_symbol = help_tooltip(
            parent=popup,
            text='Controls zoom level of the camera. Larger values zoom in, smaller values zoom out.',
        )

        grid_layout.addWidget(QLabel(trans._('Zoom:')), 1, 0)
        grid_layout.addWidget(self.zoom, 1, 1)
        grid_layout.addWidget(zoom_help_symbol, 1, 2)

    def _add_orientation_controls(
        self,
        popup: QtPopup,
        grid_layout: QGridLayout,
    ) -> None:
        """Add orientation controls to the popup."""
        orientation_widget = QWidget(popup)
        orientation_layout = QHBoxLayout()
        orientation_layout.setContentsMargins(0, 0, 0, 0)

        self.vertical_combo = enum_combobox(
            parent=popup,
            enum_class=VerticalAxisOrientation,
            current_enum=self.viewer.camera.orientation[1],
            callback=partial(
                self._update_orientation, VerticalAxisOrientation
            ),
        )

        self.horizontal_combo = enum_combobox(
            parent=popup,
            enum_class=HorizontalAxisOrientation,
            current_enum=self.viewer.camera.orientation[2],
            callback=partial(
                self._update_orientation, HorizontalAxisOrientation
            ),
        )

        if self.viewer.dims.ndisplay == 2:
            orientation_layout.addWidget(self.vertical_combo)
            orientation_layout.addWidget(self.horizontal_combo)
            self.orientation_help_symbol = help_tooltip(
                parent=popup,
                text='Controls the orientation of the vertical and horizontal camera axes.',
            )

        else:
            self.depth_combo = enum_combobox(
                parent=popup,
                enum_class=DepthAxisOrientation,
                current_enum=self.viewer.camera.orientation[0],
                callback=partial(
                    self._update_orientation, DepthAxisOrientation
                ),
            )

            orientation_layout.addWidget(self.depth_combo)
            orientation_layout.addWidget(self.vertical_combo)
            orientation_layout.addWidget(self.horizontal_combo)

            self.orientation_help_symbol = help_tooltip(
                parent=popup,
                text='',  # updated dynamically
            )
            self._update_handedness_help_symbol()
            self.viewer.camera.events.orientation.connect(
                self._update_handedness_help_symbol
            )

        orientation_widget.setLayout(orientation_layout)

        grid_layout.addWidget(QLabel(trans._('Orientation:')), 0, 0)
        grid_layout.addWidget(orientation_widget, 0, 1)
        grid_layout.addWidget(self.orientation_help_symbol, 0, 2)

    def _update_handedness_help_symbol(self, event=None) -> None:
        """Update the handedness symbol based on the camera orientation."""
        handedness = self.viewer.camera.handedness
        tooltip_text = (
            'Controls the orientation of the depth, vertical, and horizontal camera axes.\n'
            'Default is right-handed (towards, down, right).\n'
            'Default prior to 0.6.0 was left-handed (away, down, right).\n'
            f'Currently orientation is {handedness.value}-handed.'
        )
        self.orientation_help_symbol.setToolTip(tooltip_text)
        self.orientation_help_symbol.setObjectName(
            f'{handedness.value}hand_label'
        )
        self.orientation_help_symbol.style().polish(
            self.orientation_help_symbol
        )

    def open_ndisplay_camera_popup(self) -> None:
        """Show controls for camera settings based on ndisplay mode."""
        popup = QtPopup(self)
        grid_layout = QGridLayout()

        # Add orientation controls
        self._add_orientation_controls(popup, grid_layout)

        # Add shared camera controls
        self._add_shared_camera_controls(popup, grid_layout)

        # Add 3D camera controls if in 3D mode
        if self.viewer.dims.ndisplay == 3:
            self._add_3d_camera_controls(popup, grid_layout)

        popup.frame.setLayout(grid_layout)

        # Reposition popup, must be done after all widgets are added
        self._position_popup_inside_viewer(popup, self.ndisplayButton)

    def _update_orientation(
        self,
        orientation_type: type[
            DepthAxisOrientation
            | VerticalAxisOrientation
            | HorizontalAxisOrientation
        ],
        orientation_value: (
            DepthAxisOrientationStr
            | VerticalAxisOrientationStr
            | HorizontalAxisOrientationStr
        ),
    ) -> None:
        """Update the orientation of the camera.

        Parameters
        ----------
        orientation_type : type[DepthAxisOrientation | VerticalAxisOrientation | HorizontalAxisOrientation]
            The orientation type (which implies the position/axis) to update.
        value : DepthAxisOrientationStr | VerticalAxisOrientationStr | HorizontalAxisOrientationStr
            New orientation value for the updated axis.
        """
        axes = (
            DepthAxisOrientation,
            VerticalAxisOrientation,
            HorizontalAxisOrientation,
        )
        axis_to_update = axes.index(orientation_type)
        new_orientation = list(self.viewer.camera.orientation)
        new_orientation[axis_to_update] = orientation_type(orientation_value)
        self.viewer.camera.orientation = tuple(new_orientation)

    def _update_camera_angles(self, idx: int, value: float) -> None:
        """Update the camera angles.

        Parameters
        ----------
        idx : int
            Index of the angle to update. In the order of (rx, ry, rz).
        value : float
            New angle value.
        """

        angles = list(self.viewer.camera.angles)
        angles[idx] = value
        self.viewer.camera.angles = tuple(angles)

    def _update_zoom(self, value: float) -> None:
        """Update the camera zoom.

        Parameters
        ----------
        value : float
            New camera.zoom value.
        """

        self.viewer.camera.zoom = value

    def _update_perspective(self, value: float) -> None:
        """Update the camera perspective.

        Parameters
        ----------
        value : float
            New camera.perspective value.
        """

        self.viewer.camera.perspective = value

    def _open_roll_popup(self):
        """Open a grid popup to manually order the dimensions"""
        pop = QtPopup(self)

        # dims sorter widget
        dim_sorter = QtDimsSorter(self.viewer.dims, pop)
        dim_sorter.setObjectName('dim_sorter')

        # make layout
        layout = QHBoxLayout()
        layout.addWidget(dim_sorter)
        pop.frame.setLayout(layout)

        # show popup
        pop.show_above_mouse()

    def _open_grid_popup(self):
        """Open grid options pop up widget."""

        # widgets
        popup = QtPopup(self)
        grid_stride = QtSpinBox(popup)
        grid_width = QtSpinBox(popup)
        grid_height = QtSpinBox(popup)
        grid_spacing = QDoubleSpinBox(popup)
        shape_help_symbol = QtToolTipLabel(self)
        stride_help_symbol = QtToolTipLabel(self)
        spacing_help_symbol = QtToolTipLabel(self)

        shape_help_msg = trans._(
            'Number of rows and columns in the grid.\n'
            'A value of -1 for either or both of width and height will trigger an\n'
            'auto calculation of the necessary grid shape to appropriately fill\n'
            'all the layers at the appropriate stride. 0 is not a valid entry.'
        )

        stride_help_msg = trans._(
            'Number of layers to place in each grid viewbox before moving on to the next viewbox.\n'
            'A negative stride will cause the order in which the layers are placed in the grid to be reversed.\n'
            '0 is not a valid entry.'
        )

        spacing_help_msg = trans._(
            'The amount of spacing between grid viewboxes.\n'
            'If between 0 and 1, it is interpreted as a proportion of the size of the viewboxes.\n'
            'If equal or greater than 1, it is interpreted as screen pixels.'
        )

        stride_min = self.viewer.grid.__fields__['stride'].type_.ge
        stride_max = self.viewer.grid.__fields__['stride'].type_.le
        stride_not = self.viewer.grid.__fields__['stride'].type_.ne
        grid_stride.setObjectName('gridStrideBox')
        grid_stride.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_stride.setRange(stride_min, stride_max)
        grid_stride.setProhibitValue(stride_not)
        grid_stride.setValue(self.viewer.grid.stride)
        grid_stride.valueChanged.connect(self._update_grid_stride)
        self.grid_stride_box = grid_stride

        width_min = self.viewer.grid.__fields__['shape'].sub_fields[1].type_.ge
        width_not = self.viewer.grid.__fields__['shape'].sub_fields[1].type_.ne
        grid_width.setObjectName('gridWidthBox')
        grid_width.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_width.setMinimum(width_min)
        grid_width.setProhibitValue(width_not)
        grid_width.setValue(self.viewer.grid.shape[1])
        grid_width.valueChanged.connect(self._update_grid_width)
        self.grid_width_box = grid_width

        height_min = (
            self.viewer.grid.__fields__['shape'].sub_fields[0].type_.ge
        )
        height_not = (
            self.viewer.grid.__fields__['shape'].sub_fields[0].type_.ne
        )
        grid_height.setObjectName('gridStrideBox')
        grid_height.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_height.setMinimum(height_min)
        grid_height.setProhibitValue(height_not)
        grid_height.setValue(self.viewer.grid.shape[0])
        grid_height.valueChanged.connect(self._update_grid_height)
        self.grid_height_box = grid_height

        # set up spacing
        spacing_min = self.viewer.grid.__fields__['spacing'].type_.ge
        spacing_max = self.viewer.grid.__fields__['spacing'].type_.le
        spacing_step = self.viewer.grid.__fields__['spacing'].type_.step
        grid_spacing.setObjectName('gridSpacingBox')
        grid_spacing.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_spacing.setMinimum(spacing_min)
        grid_spacing.setMaximum(spacing_max)
        grid_spacing.setValue(self.viewer.grid.spacing)
        grid_spacing.setDecimals(2)
        grid_spacing.setSingleStep(spacing_step)
        grid_spacing.valueChanged.connect(self._update_grid_spacing)
        self.grid_spacing_box = grid_spacing

        # help symbols
        shape_help_symbol.setObjectName('help_label')
        shape_help_symbol.setToolTip(shape_help_msg)

        stride_help_symbol.setObjectName('help_label')
        stride_help_symbol.setToolTip(stride_help_msg)

        spacing_help_symbol.setObjectName('help_label')
        spacing_help_symbol.setToolTip(spacing_help_msg)

        # layout
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel(trans._('Grid stride:')), 0, 0)
        grid_layout.addWidget(grid_stride, 0, 1)
        grid_layout.addWidget(stride_help_symbol, 0, 2)

        grid_layout.addWidget(QLabel(trans._('Grid width:')), 1, 0)
        grid_layout.addWidget(grid_width, 1, 1)

        grid_layout.addWidget(shape_help_symbol, 1, 2, 2, 1)  # Span rows

        grid_layout.addWidget(QLabel(trans._('Grid height:')), 2, 0)
        grid_layout.addWidget(grid_height, 2, 1)

        grid_layout.addWidget(QLabel(trans._('Grid spacing:')), 3, 0)
        grid_layout.addWidget(grid_spacing, 3, 1)
        grid_layout.addWidget(spacing_help_symbol, 3, 2)

        popup.frame.setLayout(grid_layout)
        popup.show_above_mouse()

    def _update_grid_width(self, value):
        """Update the width value in grid shape.

        Parameters
        ----------
        value : int
            New grid width value.
        """

        self.viewer.grid.shape = (self.viewer.grid.shape[0], value)

    def _update_grid_stride(self, value):
        """Update stride in grid settings.

        Parameters
        ----------
        value : int
            New grid stride value.
        """

        self.viewer.grid.stride = value

    def _update_grid_height(self, value):
        """Update height value in grid shape.

        Parameters
        ----------
        value : int
            New grid height value.
        """

        self.viewer.grid.shape = (value, self.viewer.grid.shape[1])

    def _update_grid_spacing(self, value: float) -> None:
        """Update spacing value in grid settings.

        Parameters
        ----------
        value : float
            New grid spacing value.
        """
        self.viewer.grid.spacing = value


def _omit_viewer_args(constructor):
    @wraps(constructor)
    def _func(*args, **kwargs):
        if len(args) > 1 and not isinstance(args[1], str):
            warnings.warn(
                trans._(
                    'viewer argument is deprecated since 0.4.14 and should not be used'
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            args = args[:1] + args[2:]
        if 'viewer' in kwargs:
            warnings.warn(
                trans._(
                    'viewer argument is deprecated since 0.4.14 and should not be used'
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            del kwargs['viewer']
        return constructor(*args, **kwargs)

    return _func


class QtViewerPushButton(QPushButton):
    """Push button.

    Parameters
    ----------
    button_name : str
        Name of button.
    tooltip : str
        Tooltip for button. If empty then `button_name` is used
    slot : Callable, optional
        callable to be triggered on button click
    action : str
        action name to be triggered on button click
    """

    @_omit_viewer_args
    def __init__(
        self,
        button_name: str,
        tooltip: str = '',
        slot=None,
        action: str = '',
        extra_tooltip_text: str = '',
    ) -> None:
        super().__init__()

        self.setToolTip(tooltip or button_name)
        self.setProperty('mode', button_name)
        if slot is not None:
            self.clicked.connect(slot)
        if action:
            action_manager.bind_button(
                action, self, extra_tooltip_text=extra_tooltip_text
            )
