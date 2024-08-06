import warnings
from functools import wraps
from typing import TYPE_CHECKING

from app_model.expressions import get_context
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from napari._app_model.constants._menus import MenuId
from napari._qt._qapp_model import build_qmodel_menu, build_qmodel_toolbar
from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.widgets.qt_dims_sorter import QtDimsSorter
from napari._qt.widgets.qt_spinbox import QtSpinBox
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.utils.action_manager import action_manager
from napari.utils.translations import trans

if TYPE_CHECKING:
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
    deleteButton : qtpy.QtWidgets.QToolButton
        Button to delete selected layers.
    newLabelsButton : qtpy.QtWidgets.QToolButton
        Button to add new Label layer.
    newPointsButton : qtpy.QtWidgets.QToolButton
        Button to add new Points layer.
    newShapesButton : qtpy.QtWidgets.QToolButton
        Button to add new Shapes layer.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer: 'ViewerModel') -> None:
        super().__init__()

        self.viewer = viewer

        # Setup toolbar
        self._menu = build_qmodel_menu(
            MenuId.VIEWER_NEW_DELETE_LAYER, parent=self
        )
        self.toolbar = build_qmodel_toolbar(
            MenuId.VIEWER_NEW_DELETE_LAYER,
            title='Layer creation controls',
            parent=self,
        )

        # TODO: Insert empty widget/spacer after new labels button/before delete button
        # Setup controls/buttons
        new_points_action = self._menu.findAction(
            'napari.viewer.new_layer.new_points'
        )
        new_points_tool = self.toolbar.widgetForAction(new_points_action)
        new_points_tool.setProperty('mode', 'new_points')
        self.newPointsButton = new_points_tool

        new_shapes_action = self._menu.findAction(
            'napari.viewer.new_layer.new_shapes'
        )
        new_shapes_tool = self.toolbar.widgetForAction(new_shapes_action)
        new_shapes_tool.setProperty('mode', 'new_shapes')
        self.newShapesButton = new_shapes_tool

        new_labels_action = self._menu.findAction(
            'napari.viewer.new_layer.new_labels'
        )
        new_labels_tool = self.toolbar.widgetForAction(new_labels_action)
        new_labels_tool.setProperty('mode', 'new_labels')
        self.newLabelsButton = new_labels_tool

        delete_action = self._menu.findAction(
            'napari.viewer.delete_selected_layers'
        )
        delete_tool = self.toolbar.widgetForAction(delete_action)
        delete_tool.setProperty('mode', 'delete_button')
        self.deleteButton = delete_tool

        empty_widget = QWidget(self)
        empty_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self.insertWidget(delete_action, empty_widget)

        # Setup layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def insertWidget(self, before, widget):
        return self.toolbar.insertWidget(before, widget)


class QtViewerButtons(QFrame):
    """Button controls for the napari viewer.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    consoleButton : qtpy.QtWidgets.QToolButton
        Button to open IPython console within napari.
    rollDimsButton : qtpy.QtWidgets.QToolButton
        Button to roll orientation of spatial dimensions in the napari viewer.
    transposeDimsButton : qtpy.QtWidgets.QToolButton
        Button to transpose dimensions in the napari viewer.
    resetViewButton : qtpy.QtWidgets.QToolButton
        Button resetting the view of the rendered scene.
    gridViewButton : qtpy.QtWidgets.QToolButton
        Button to toggle grid view mode of layers on and off.
    ndisplayButton : qtpy.QtWidgets.QToolButton
        Button to toggle number of displayed dimensions.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    toolbar : app_model.backends.qt import QModelToolBar
        app-model toolbar that gets populated with viewer menu actions.
    """

    def __init__(self, viewer: 'ViewerModel') -> None:
        super().__init__()

        # General attributes
        self.viewer = viewer

        # Toolbar attributes
        self._menu = build_qmodel_menu(MenuId.VIEWER_CONTROLS, parent=self)
        self.toolbar = build_qmodel_toolbar(
            MenuId.VIEWER_CONTROLS, title='Viewer controls', parent=self
        )

        @self.viewer.events.update_ctx.connect
        def _update_toolbar_from_context(event):
            ctx = get_context(event.source)
            self.toolbar.update_from_context(ctx)

        # Setup controls/buttons
        console_action = self._menu.findAction(
            'napari.viewer.toggle_console_visibility'
        )
        console_tool = self.toolbar.widgetForAction(console_action)
        console_tool.setProperty('mode', 'console')
        console_tool.setProperty('expanded', False)
        self.consoleButton = console_tool

        roll_action = self._menu.findAction('napari.viewer.roll_axes')
        roll_tool = self.toolbar.widgetForAction(roll_action)
        roll_tool.setProperty('mode', 'roll')
        roll_tool.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        roll_tool.customContextMenuRequested.connect(self._open_roll_popup)
        self.rollDimsButton = roll_tool

        transpose_action = self._menu.findAction(
            'napari.viewer.transpose_axes'
        )
        transpose_tool = self.toolbar.widgetForAction(transpose_action)
        transpose_tool.setProperty('mode', 'transpose')
        self.transposeDimsButton = transpose_tool

        reset_view_action = self._menu.findAction('napari.viewer.reset_view')
        reset_view_tool = self.toolbar.widgetForAction(reset_view_action)
        reset_view_tool.setProperty('mode', 'home')
        self.resetViewButton = reset_view_tool

        grid_view_action = self._menu.findAction('napari.viewer.toggle_grid')
        grid_view_tool = self.toolbar.widgetForAction(grid_view_action)
        grid_view_tool.setProperty('mode', 'grid_view_button')
        grid_view_tool.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        grid_view_tool.customContextMenuRequested.connect(
            self._open_grid_popup
        )
        self.gridViewButton = grid_view_tool

        ndisplay_action = self._menu.findAction(
            'napari.viewer.toggle_ndisplay'
        )
        ndisplay_tool = self.toolbar.widgetForAction(ndisplay_action)
        ndisplay_tool.setProperty('mode', 'ndisplay_button')
        ndisplay_tool.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        ndisplay_tool.customContextMenuRequested.connect(
            self.open_perspective_popup
        )
        self.ndisplayButton = ndisplay_tool

        # Setup layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def open_perspective_popup(self):
        """Show a slider to control the viewer `camera.perspective`."""
        if self.viewer.dims.ndisplay != 3:
            return

        # make slider connected to perspective parameter
        sld = QSlider(Qt.Orientation.Horizontal, self)
        sld.setRange(0, max(90, int(self.viewer.camera.perspective)))
        sld.setValue(int(self.viewer.camera.perspective))
        sld.valueChanged.connect(
            lambda v: setattr(self.viewer.camera, 'perspective', v)
        )
        self.perspective_slider = sld

        # make layout
        layout = QHBoxLayout()
        layout.addWidget(QLabel(trans._('Perspective'), self))
        layout.addWidget(sld)

        # popup and show
        pop = QtPopup(self)
        pop.frame.setLayout(layout)
        pop.show_above_mouse()

    def _open_roll_popup(self):
        """Open a grid popup to manually order the dimensions"""
        if self.viewer.dims.ndisplay != 2:
            return

        # popup
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
        shape_help_symbol = QtToolTipLabel(self)
        stride_help_symbol = QtToolTipLabel(self)
        blank = QLabel(self)  # helps with placing help symbols.

        shape_help_msg = trans._(
            'Number of rows and columns in the grid. A value of -1 for either or both of width and height will trigger an auto calculation of the necessary grid shape to appropriately fill all the layers at the appropriate stride. 0 is not a valid entry.'
        )

        stride_help_msg = trans._(
            'Number of layers to place in each grid square before moving on to the next square. The default ordering is to place the most visible layer in the top left corner of the grid. A negative stride will cause the order in which the layers are placed in the grid to be reversed. 0 is not a valid entry.'
        )

        # set up
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

        shape_help_symbol.setObjectName('help_label')
        shape_help_symbol.setToolTip(shape_help_msg)

        stride_help_symbol.setObjectName('help_label')
        stride_help_symbol.setToolTip(stride_help_msg)

        # layout
        form_layout = QFormLayout()
        form_layout.insertRow(0, QLabel(trans._('Grid stride:')), grid_stride)
        form_layout.insertRow(1, QLabel(trans._('Grid width:')), grid_width)
        form_layout.insertRow(2, QLabel(trans._('Grid height:')), grid_height)

        help_layout = QVBoxLayout()
        help_layout.addWidget(stride_help_symbol)
        help_layout.addWidget(blank)
        help_layout.addWidget(shape_help_symbol)

        layout = QHBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(help_layout)

        popup.frame.setLayout(layout)

        popup.show_above_mouse()

        # adjust placement of shape help symbol.  Must be done last
        # in order for this movement to happen.
        delta_x = 0
        delta_y = -15
        shape_pos = (
            shape_help_symbol.x() + delta_x,
            shape_help_symbol.y() + delta_y,
        )
        shape_help_symbol.move(QPoint(*shape_pos))

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
        self, button_name: str, tooltip: str = '', slot=None, action: str = ''
    ) -> None:
        super().__init__()

        self.setToolTip(tooltip or button_name)
        self.setProperty('mode', button_name)
        if slot is not None:
            self.clicked.connect(slot)
        if action:
            action_manager.bind_button(action, self)
