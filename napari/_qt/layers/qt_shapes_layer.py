from collections.abc import Iterable
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QLabel,
    QComboBox,
    QSlider,
    QFrame,
    QGridLayout,
    QHBoxLayout,
)
from vispy.color import Color
from .qt_base_layer import QtLayerControls
from ...layers.shapes._shapes_constants import Mode
from ..qt_mode_buttons import QtModeRadioButton, QtModePushButton
from ..utils import disable_with_opacity


class QtShapesControls(QtLayerControls):
    """Qt view and controls for the napari Shapes layer.

    Parameters
    ----------
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.

    Attributes
    ----------
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for shapes layer modes
        (SELECT, DIRECT, PAN_ZOOM, ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE,
        ADD_PATH, ADD_POLYGON, VERTEX_INSERT, VERTEX_REMOVE).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected shapes
    direct_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select individual vertices in shapes.
    edgeColorSwatch : qtpy.QtWidgets.QFrame
        Thumbnail display of points edge color.
    edgeComboBox : qtpy.QtWidgets.QComboBox
        Drop down list allowing user to set edge color of points.
    ellipse_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add ellipses to shapes layer.
    faceColorSwatch : qtpy.QtWidgets.QFrame
        Thumbnail display of points face color.
    faceComboBox : qtpy.QtWidgets.QComboBox
        Drop down list allowing user to set face color of points.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.
    line_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add lines to shapes layer.
    move_back_button : qtpy.QtWidgets.QtModePushButton
        Button to move selected shape(s) to the back.
    move_front_button : qtpy.QtWidgets.QtModePushButton
        Button to move shape(s) to the front.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button to pan/zoom shapes layer.
    path_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add paths to shapes layer.
    polygon_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add polygons to shapes layer.
    rectangle_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add rectangles to shapes layer.
    select_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select shapes.
    vertex_insert_button : qtpy.QtWidgets.QtModeRadioButton
        Button to insert vertex into shape.
    vertex_remove_button : qtpy.QtWidgets.QtModeRadioButton
        Button to remove vertex from shapes.
    widthSlider : qtpy.QtWidgets.QSlider
        Slider controlling line edge width of shapes.

    Raises
    ------
    ValueError
        Raise error if shapes mode is not recognized.
    """

    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)
        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.edge_color.connect(self._on_edge_color_change)
        self.layer.events.face_color.connect(self._on_face_color_change)
        self.layer.events.editable.connect(self._on_editable_change)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.current_edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeWidth)
        self.widthSlider = sld

        face_comboBox = QComboBox()
        face_comboBox.addItems(self.layer._colors)
        face_comboBox.activated[str].connect(self.changeFaceColor)
        self.faceComboBox = face_comboBox
        self.faceColorSwatch = QFrame()
        self.faceColorSwatch.setObjectName('colorSwatch')
        self.faceColorSwatch.setToolTip('Face color swatch')
        self._on_face_color_change()

        edge_comboBox = QComboBox()
        edge_comboBox.addItems(self.layer._colors)
        edge_comboBox.activated[str].connect(self.changeEdgeColor)
        self.edgeComboBox = edge_comboBox
        self.edgeColorSwatch = QFrame()
        self.edgeColorSwatch.setObjectName('colorSwatch')
        self.edgeColorSwatch.setToolTip('Edge color swatch')
        self._on_edge_color_change()

        self.select_button = QtModeRadioButton(
            layer, 'select', Mode.SELECT, tooltip='Select shapes'
        )
        self.direct_button = QtModeRadioButton(
            layer, 'direct', Mode.DIRECT, tooltip='Select vertices'
        )
        self.panzoom_button = QtModeRadioButton(
            layer, 'zoom', Mode.PAN_ZOOM, tooltip='Pan/zoom', checked=True
        )
        self.rectangle_button = QtModeRadioButton(
            layer, 'rectangle', Mode.ADD_RECTANGLE, tooltip='Add rectangles'
        )
        self.ellipse_button = QtModeRadioButton(
            layer, 'ellipse', Mode.ADD_ELLIPSE, tooltip='Add ellipses'
        )
        self.line_button = QtModeRadioButton(
            layer, 'line', Mode.ADD_LINE, tooltip='Add lines'
        )
        self.path_button = QtModeRadioButton(
            layer, 'path', Mode.ADD_PATH, tooltip='Add paths'
        )
        self.polygon_button = QtModeRadioButton(
            layer, 'polygon', Mode.ADD_POLYGON, tooltip='Add polygons'
        )
        self.vertex_insert_button = QtModeRadioButton(
            layer, 'vertex_insert', Mode.VERTEX_INSERT, tooltip='Insert vertex'
        )
        self.vertex_remove_button = QtModeRadioButton(
            layer, 'vertex_remove', Mode.VERTEX_REMOVE, tooltip='Remove vertex'
        )

        self.move_front_button = QtModePushButton(
            layer,
            'move_front',
            slot=self.layer.move_to_front,
            tooltip='Move to front',
        )
        self.move_back_button = QtModePushButton(
            layer,
            'move_back',
            slot=self.layer.move_to_back,
            tooltip='Move to back',
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip='Delete selected shapes',
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.direct_button)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.rectangle_button)
        self.button_group.addButton(self.ellipse_button)
        self.button_group.addButton(self.line_button)
        self.button_group.addButton(self.path_button)
        self.button_group.addButton(self.polygon_button)
        self.button_group.addButton(self.vertex_insert_button)
        self.button_group.addButton(self.vertex_remove_button)

        button_grid = QGridLayout()
        button_grid.addWidget(self.vertex_remove_button, 0, 2)
        button_grid.addWidget(self.vertex_insert_button, 0, 3)
        button_grid.addWidget(self.delete_button, 0, 4)
        button_grid.addWidget(self.direct_button, 0, 5)
        button_grid.addWidget(self.select_button, 0, 6)
        button_grid.addWidget(self.panzoom_button, 0, 7)
        button_grid.addWidget(self.move_back_button, 1, 1)
        button_grid.addWidget(self.move_front_button, 1, 2)
        button_grid.addWidget(self.ellipse_button, 1, 3)
        button_grid.addWidget(self.rectangle_button, 1, 4)
        button_grid.addWidget(self.polygon_button, 1, 5)
        button_grid.addWidget(self.line_button, 1, 6)
        button_grid.addWidget(self.path_button, 1, 7)
        button_grid.setContentsMargins(5, 0, 0, 5)
        button_grid.setColumnStretch(0, 1)
        button_grid.setSpacing(4)

        face_color_layout = QHBoxLayout()
        face_color_layout.addWidget(self.faceColorSwatch)
        face_color_layout.addWidget(self.faceComboBox)
        edge_color_layout = QHBoxLayout()
        edge_color_layout.addWidget(self.edgeColorSwatch)
        edge_color_layout.addWidget(self.edgeComboBox)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addLayout(button_grid, 0, 0, 1, 2)
        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0)
        self.grid_layout.addWidget(self.opacitySlider, 1, 1)
        self.grid_layout.addWidget(QLabel('edge width:'), 2, 0)
        self.grid_layout.addWidget(self.widthSlider, 2, 1)
        self.grid_layout.addWidget(QLabel('blending:'), 3, 0)
        self.grid_layout.addWidget(self.blendComboBox, 3, 1)
        self.grid_layout.addWidget(QLabel('face color:'), 4, 0)
        self.grid_layout.addLayout(face_color_layout, 4, 1)
        self.grid_layout.addWidget(QLabel('edge color:'), 5, 0)
        self.grid_layout.addLayout(edge_color_layout, 5, 1)
        self.grid_layout.setRowStretch(6, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def mouseMoveEvent(self, event):
        """On mouse move, update layer mode status.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.layer.status = str(self.layer.mode)

    def set_mode(self, event):
        """"Update ticks in checkbox widgets when shapes layer mode changed.

        Available modes for shapes layer are:
        * SELECT
        * DIRECT
        * PAN_ZOOM
        * ADD_RECTANGLE
        * ADD_ELLIPSE
        * ADD_LINE
        * ADD_PATH
        * ADD_POLYGON
        * VERTEX_INSERT
        * VERTEX_REMOVE

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.

        Raises
        ------
        ValueError
            Raise error if event.mode is not ADD, PAN_ZOOM, or SELECT.
        """
        mode_buttons = {
            Mode.SELECT: self.select_button,
            Mode.DIRECT: self.direct_button,
            Mode.PAN_ZOOM: self.panzoom_button,
            Mode.ADD_RECTANGLE: self.rectangle_button,
            Mode.ADD_ELLIPSE: self.ellipse_button,
            Mode.ADD_LINE: self.line_button,
            Mode.ADD_PATH: self.path_button,
            Mode.ADD_POLYGON: self.polygon_button,
            Mode.VERTEX_INSERT: self.vertex_insert_button,
            Mode.VERTEX_REMOVE: self.vertex_remove_button,
        }

        if event.mode in mode_buttons:
            mode_buttons[event.mode].setChecked(True)
        else:
            raise ValueError(f"Mode '{event.mode}'not recognized")

    def changeFaceColor(self, text):
        """Change face color of shapes.

        Parameters
        ----------
        text : str
            Face color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        self.layer.current_face_color = text

    def changeEdgeColor(self, text):
        """Change edge color of shapes.

        Parameters
        ----------
        text : str
            Edge color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        self.layer.current_edge_color = text

    def changeWidth(self, value):
        """Change edge line width of shapes on the layer model.

        Parameters
        ----------
        value : float
            Line width of shapes.
        """
        self.layer.current_edge_width = float(value) / 2

    def changeOpacity(self, value):
        """Change opacity value of shapes on the layer model.

        Parameters
        ----------
        value : float
            Opacity value for shapes.
            Input range 0 - 100 (transparent to fully opaque).
        """
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.current_opacity = value / 100

    def _on_edge_width_change(self, event=None):
        """Receive layer model edge line width change event and update slider.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional.
            Event from the Qt context, by default None.
        """
        with self.layer.events.edge_width.blocker():
            value = self.layer.current_edge_width
            value = np.clip(int(2 * value), 0, 40)
            self.widthSlider.setValue(value)

    def _on_edge_color_change(self, event=None):
        """Receive layer model edge color change event and update color swatch.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional.
            Event from the Qt context, by default None.
        """
        with self.layer.events.edge_color.blocker():
            index = self.edgeComboBox.findText(
                self.layer.current_edge_color, Qt.MatchFixedString
            )
            self.edgeComboBox.setCurrentIndex(index)
        color = Color(self.layer.current_edge_color).hex
        self.edgeColorSwatch.setStyleSheet("background-color: " + color)

    def _on_face_color_change(self, event=None):
        """Receive layer model face color change event and update color swatch.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional.
            Event from the Qt context, by default None.
        """
        with self.layer.events.face_color.blocker():
            index = self.faceComboBox.findText(
                self.layer.current_face_color, Qt.MatchFixedString
            )
            self.faceComboBox.setCurrentIndex(index)
        color = Color(self.layer.current_face_color).hex
        self.faceColorSwatch.setStyleSheet("background-color: " + color)

    def _on_opacity_change(self, event=None):
        """Receive layer model opacity change event and update opacity slider.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional.
            Event from the Qt context, by default None.
        """
        with self.layer.events.opacity.blocker():
            self.opacitySlider.setValue(self.layer.current_opacity * 100)

    def _on_editable_change(self, event=None):
        """Receive layer model editable change event & enable/disable buttons.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent, optional.
            Event from the Qt context, by default None.
        """
        disable_with_opacity(
            self,
            [
                'select_button',
                'direct_button',
                'rectangle_button',
                'ellipse_button',
                'line_button',
                'path_button',
                'polygon_button',
                'vertex_remove_button',
                'vertex_insert_button',
                'delete_button',
                'move_back_button',
                'move_front_button',
            ],
            self.layer.editable,
        )
