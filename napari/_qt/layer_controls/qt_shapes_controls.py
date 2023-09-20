from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import NewQtLayerControls
from napari._qt.layer_controls.widgets import (
    QtEdgeColorControl,
    QtEdgeWidthSliderControl,
    QtFaceColorControl,
    QtOpacityBlendingControls,
    QtTextVisibilityControl,
)
from napari.layers.shapes._shapes_constants import Mode
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtShapesControls(NewQtLayerControls):
    """Qt view and controls for the napari Shapes layer.

    Parameters
    ----------
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.

    Mode buttons
    ------------
    delete_button : napari._qt.widgets.qt_mode_buttons.QtModePushButton
        Button to delete selected shapes
    direct_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to select individual vertices in shapes.
    ellipse_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to add ellipses to shapes layer.
    line_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to add lines to shapes layer.
    move_back_button : napari._qt.widgets.qt_mode_buttons.QtModePushButton
        Button to move selected shape(s) to the back.
    move_front_button : napari._qt.widgets.qt_mode_buttons.QtModePushButton
        Button to move shape(s) to the front.
    panzoom_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to pan/zoom shapes layer.
    path_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to add paths to shapes layer.
    polygon_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to add polygons to shapes layer.
    polygon_lasso_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to add polygons to shapes layer with a lasso tool.
    rectangle_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to add rectangles to shapes layer.
    select_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to select shapes.
    vertex_insert_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to insert vertex into shape.
    vertex_remove_button : napari._qt.widgets.qt_mode_buttons.QtModeRadioButton
        Button to remove vertex from shapes.

    Raises
    ------
    ValueError
        Raise error if shapes mode is not recognized.
    """

    _layer: 'napari.layers.Shapes'

    def __init__(self, layer: 'napari.layers.Shapes') -> None:
        # Shapes Mode enum counts with the following modes:
        #    SELECT, DIRECT, PAN_ZOOM, ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE,
        #    ADD_PATH, ADD_POLYGON, VERTEX_INSERT, VERTEX_REMOVE
        super().__init__(layer, mode_options=Mode)

        # Setup mode buttons
        # panzoom_button
        self._add_radio_button_mode(
            'pan_zoom',
            Mode.PAN_ZOOM,
            "activate_shapes_pan_zoom_mode",
            0,
            5,
            extra_tooltip_text=trans._('(or hold Space)'),
            edit_button=False,
            checked=True,
        )
        # path_button
        self._add_radio_button_mode(
            'path', Mode.ADD_PATH, "activate_add_path_mode", 1, 6
        )
        # line_button
        self._add_radio_button_mode(
            'line', Mode.ADD_LINE, "activate_add_line_mode", 1, 5
        )
        # polygon_button
        self._add_radio_button_mode(
            'polygon',
            Mode.ADD_POLYGON,
            "activate_add_polygon_mode",
            1,
            3,
        )
        # polygon_lasso_button
        self._add_radio_button_mode(
            'polygon_lasso',
            Mode.ADD_POLYGON_LASSO,
            "activate_add_polygon_lasso_mode",
            1,
            4,
        )
        # rectangle_button
        self._add_radio_button_mode(
            'rectangle',
            Mode.ADD_RECTANGLE,
            "activate_add_rectangle_mode",
            1,
            2,
        )
        # ellipse_button
        self._add_radio_button_mode(
            'ellipse',
            Mode.ADD_ELLIPSE,
            "activate_add_ellipse_mode",
            1,
            1,
        )
        # select_button
        self._add_radio_button_mode(
            'select', Mode.SELECT, "activate_select_mode", 0, 4
        )
        # direct_button
        self._add_radio_button_mode(
            'direct', Mode.DIRECT, "activate_direct_mode", 0, 3
        )
        # delete_button
        self._add_push_button_action(
            'delete_shape',
            0,
            2,
            slot=self._layer.remove_selected,
            tooltip=trans._(
                "Delete selected shapes ({shortcut})",
                shortcut=Shortcut('Backspace').platform,
            ),
        )
        # vertex_insert_button
        self._add_radio_button_mode(
            'vertex_insert',
            Mode.VERTEX_INSERT,
            "activate_vertex_insert_mode",
            0,
            1,
        )
        # vertex_remove_button
        self._add_radio_button_mode(
            'vertex_remove',
            Mode.VERTEX_REMOVE,
            "activate_vertex_remove_mode",
            0,
            0,
        )
        # move_front_button
        self._add_push_button_action(
            'move_front',
            1,
            0,
            action_name='move_shapes_selection_to_front',
            slot=self._layer.move_to_front,
            tooltip=trans._('Move to front'),
        )
        # move_back_button
        self._add_push_button_action(
            'move_back',
            0,
            6,
            action_name='move_shapes_selection_to_back',
            slot=self._layer.move_to_back,
            tooltip=trans._('Move to back'),
        )
        self._on_editable_or_visible_change()

        # Setup widget controls
        # TODO: Should be done when instantiating layer controls class via some
        # sort of mapping between layer attributes and QObject classes
        # with QWidgets-Layer atts connection logic
        self.add_display_widget_controls(
            QtOpacityBlendingControls(self, layer)
        )
        self.add_display_widget_controls(QtEdgeWidthSliderControl(self, layer))
        self.add_display_widget_controls(QtEdgeColorControl(self, layer))
        self.add_display_widget_controls(QtFaceColorControl(self, layer))
        self.add_display_widget_controls(QtTextVisibilityControl(self, layer))

    def _on_ndisplay_changed(self) -> None:
        self._layer.editable = self.ndisplay == 2
