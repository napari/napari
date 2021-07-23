from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
)

from ...utils.action_manager import action_manager
from ...utils.interactions import Shortcut
from ...utils.translations import trans
from ..dialogs.qt_modal import QtPopup


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

    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.deleteButton = QtDeleteButton(self.viewer)
        self.newPointsButton = QtViewerPushButton(
            self.viewer,
            'new_points',
            trans._('New points layer'),
            lambda: self.viewer.add_points(
                ndim=max(self.viewer.dims.ndim, 2),
                scale=self.viewer.layers.extent.step,
            ),
        )

        self.newShapesButton = QtViewerPushButton(
            self.viewer,
            'new_shapes',
            trans._('New shapes layer'),
            lambda: self.viewer.add_shapes(
                ndim=max(self.viewer.dims.ndim, 2),
                scale=self.viewer.layers.extent.step,
            ),
        )
        self.newLabelsButton = QtViewerPushButton(
            self.viewer,
            'new_labels',
            trans._('New labels layer'),
            lambda: self.viewer._new_labels(),
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.newPointsButton)
        layout.addWidget(self.newShapesButton)
        layout.addWidget(self.newLabelsButton)
        layout.addStretch(0)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)


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
    gridViewButton : QtStateButton
        Button to toggle grid view mode of layers on and off.
    ndisplayButton : QtStateButton
        Button to toggle number of displayed dimensions.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        action_manager.context['viewer'] = viewer

        # self._set_grid_options()

        def active_layer():
            if len(self.viewer.layers.selection) == 1:
                return next(iter(self.viewer.layers.selection))
            else:
                return None

        action_manager.context['layer'] = active_layer

        self.consoleButton = QtViewerPushButton(
            self.viewer,
            'console',
            trans._(
                "Open IPython terminal",
            ),
        )
        self.consoleButton.setProperty('expanded', False)
        self.rollDimsButton = QtViewerPushButton(
            self.viewer,
            'roll',
        )

        action_manager.bind_button('napari:roll_axes', self.rollDimsButton)

        self.transposeDimsButton = QtViewerPushButton(
            self.viewer,
            'transpose',
        )

        action_manager.bind_button(
            'napari:transpose_axes', self.transposeDimsButton
        )

        self.resetViewButton = QtViewerPushButton(self.viewer, 'home')
        action_manager.bind_button('napari:reset_view', self.resetViewButton)

        self.gridViewButton = QtStateButton(
            'grid_view_button',
            self.viewer.grid,
            'enabled',
            self.viewer.grid.events,
        )

        self.gridViewButton.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gridViewButton.customContextMenuRequested.connect(
            self.open_grid_popup
        )
        # look here for button!!!  Need to copy the customcontext stuff you see below to try to get the
        # menu to open. need to change your function at bottom to just be about the pop up and put the
        # button back in place.  see the ndisplay button for example on how to do it.
        action_manager.bind_button('napari:toggle_grid', self.gridViewButton)

        self.ndisplayButton = QtStateButton(
            "ndisplay_button",
            self.viewer.dims,
            'ndisplay',
            self.viewer.dims.events.ndisplay,
            2,
            3,
        )
        self.ndisplayButton.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ndisplayButton.customContextMenuRequested.connect(
            self.open_perspective_popup
        )
        action_manager.bind_button(
            'napari:toggle_ndisplay', self.ndisplayButton
        )

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

    def open_perspective_popup(self):
        """Show a slider to control the viewer `camera.perspective`."""
        if self.viewer.dims.ndisplay != 3:
            return

        # make slider connected to perspective parameter
        sld = QSlider(Qt.Horizontal, self)
        sld.setRange(0, max(90, self.viewer.camera.perspective))
        sld.setValue(self.viewer.camera.perspective)
        sld.valueChanged.connect(
            lambda v: setattr(self.viewer.camera, 'perspective', v)
        )

        # make layout
        layout = QHBoxLayout()
        layout.addWidget(QLabel(trans._('Perspective'), self))
        layout.addWidget(sld)

        # popup and show
        pop = QtPopup(self)
        pop.frame.setLayout(layout)
        pop.show_above_mouse()

    def open_grid_popup(self):
        """Open grid options pop up."""

        # layouts

        popup = QtPopup(self)
        form_layout = QFormLayout()

        grid_stride = QSpinBox(popup)
        grid_stride.setObjectName("griddStrideBox")
        grid_stride.setAlignment(Qt.AlignCenter)
        grid_stride.setMaximum(20)
        grid_stride.setMinimum(-20)
        grid_stride.setValue(self.viewer.grid.stride)
        grid_stride.valueChanged.connect(self._update_grid_stride)
        form_layout.insertRow(
            0,
            QLabel(trans._('Grid Stride:'), parent=popup),
            grid_stride,
        )
        self.grid_stride_box = grid_stride

        wh_values = list(range(-1, 10))
        wh_values = wh_values.pop(1)

        grid_width = QSpinBox(popup)
        grid_width.setObjectName("griddWidthBox")
        grid_width.setAlignment(Qt.AlignCenter)
        grid_width.setMaximum(20)
        grid_width.setMinimum(-1)
        grid_width.setValue(self.viewer.grid.shape[1])
        grid_width.valueChanged.connect(self._update_grid_width)

        form_layout.insertRow(
            1,
            QLabel(trans._('Grid Width:'), parent=popup),
            grid_width,
        )
        self.grid_width_box = grid_width

        grid_height = QSpinBox(popup)
        grid_height.setObjectName("gdStrideBox")
        grid_height.setAlignment(Qt.AlignCenter)
        grid_height.setMaximum(20)
        grid_height.setMinimum(-20)
        grid_height.setValue(self.viewer.grid.shape[0])
        grid_height.valueChanged.connect(self._update_grid_height)

        form_layout.insertRow(
            2,
            QLabel(trans._('Grid Height:'), parent=popup),
            grid_height,
        )
        self.grid_height_box = grid_height

        help_layout = QGridLayout()

        help_symbol = QLabel(self)
        help_symbol.setObjectName(
            "help_label"
        )  # need to change with proper symbol
        help_symbol.setToolTip('Testing tool tip-yay, its here!!!')

        self.shape_help_msg = QLabel(
            trans._(
                'Number of rows and columns in the grid. A value of -1 for either or both of will be used the row and column numbers will trigger an auto calculation of the necessary grid shape to appropriately fill all the layers at the appropriate stride.'
            )
        )

        #         QString toolTip = QString("<FONT COLOR=black>");
        # toolTip += ("I am the very model of a modern major general, I've information vegetable animal and mineral, I know the kinges of England and I quote the fights historical from Marathon to Waterloo in order categorical...");
        # toolTip += QString("</FONT>");
        # widget->setToolTip(sToolTip);

        txt = f"<FONT> {self.shape_help_msg.text()}</FONT>"
        print(txt)
        self.shape_help_msg.setWordWrap(True)
        help_symbol2 = QLabel(self)
        help_symbol2.setObjectName(
            "help_label"
        )  # need to change with proper symbol
        help_symbol2.setToolTip(txt)
        # help_symbol2.setToolTip('Testing tool tip-yay, its here!!!')
        # help_symbol2.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        # help_symbol2.clicked.connect(self._show_help_message)

        blank = QLabel(self)

        help_layout.addWidget(help_symbol, 0, 0)
        help_layout.addWidget(blank, 2, 0)
        help_layout.addWidget(help_symbol2, 3, 0)

        layout = QHBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(help_layout)
        popup.frame.setLayout(layout)
        popup.show_above_mouse()

    def _show_help_message(self):
        from napari._qt.widgets.qt_keyboard_settings import KeyBindWarnPopup

        # delta_y = 105
        # delta_x = 10
        # # global_point = self.mapToGlobal(
        #     QPoint(
        #         self.
        #         self._table.columnViewportPosition(self._shortcut_col)
        #         + delta_x,
        #         self._table.rowViewportPosition(row) + delta_y,
        #     )
        # )

        dlg = KeyBindWarnPopup(
            text=self.shape_help_msg,
        )
        # self._warn_dialog.move(global_point)

        # Styling adjustments.
        dlg.resize(400, dlg.sizeHint().height())

        # dlg._message.resize(
        #     200, dlg._message.sizeHint().height()
        # )

        dlg.exec_()
        print('pushed!')

    def _update_grid_width(self, value):
        self.viewer.grid.shape = (self.viewer.grid.shape[0], value)

    def _update_grid_stride(self, value):
        self.viewer.grid.stride = value

    def _update_grid_height(self, value):
        self.viewer.grid.shape = (value, self.viewer.grid.shape[1])


class QtDeleteButton(QPushButton):
    """Delete button to remove selected layers.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    hover : bool
        Hover is true while mouse cursor is on the button widget.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setToolTip(
            trans._(
                "Delete selected layers ({shortcut})",
                shortcut=Shortcut("Control-Backspace"),
            )
        )
        self.setAcceptDrops(True)
        self.clicked.connect(lambda: self.viewer.layers.remove_selected())

    def dragEnterEvent(self, event):
        """The cursor enters the widget during a drag and drop operation.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.accept()
        self.hover = True
        self.update()

    def dragLeaveEvent(self, event):
        """The cursor leaves the widget during a drag and drop operation.

        Using event.ignore() here allows the event to pass through the
        parent widget to its child widget, otherwise the parent widget
        would catch the event and not pass it on to the child widget.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.ignore()
        self.hover = False
        self.update()

    def dropEvent(self, event):
        """The drag and drop mouse event is completed.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        event.accept()
        layer_name = event.mimeData().text()
        layer = self.viewer.layers[layer_name]
        if not layer.selected:
            self.viewer.layers.remove(layer)
        else:
            self.viewer.layers.remove_selected()


class QtViewerPushButton(QPushButton):
    """Push button.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer, button_name, tooltip=None, slot=None):
        super().__init__()

        self.viewer = viewer
        self.setToolTip(tooltip or button_name)
        self.setProperty('mode', button_name)
        if slot is not None:
            self.clicked.connect(slot)


class QtStateButton(QtViewerPushButton):
    """Button to toggle between two states.

    Parameters
    ----------
    button_name : str
        A string that will be used in qss to style the button with the
        QtStateButton[mode=...] selector,
    target : object
        object on which you want to change the property when button pressed.
    attribute:
        name of attribute on `object` you wish to change.
    events: EventEmitter
        event emitter that will trigger when value is changed
    onstate: Any
        value to use for ``setattr(object, attribute, onstate)`` when clicking
        this button
    offstate: Any
        value to use for ``setattr(object, attribute, offstate)`` when clicking
        this button.
    """

    def __init__(
        self,
        button_name,
        target,
        attribute,
        events,
        onstate=True,
        offstate=False,
    ):
        super().__init__(target, button_name)
        self.setCheckable(True)

        self._target = target
        self._attribute = attribute
        self._onstate = onstate
        self._offstate = offstate
        self._events = events
        self._events.connect(self._on_change)
        self.clicked.connect(self.change)
        self._on_change()

    def change(self):
        """Toggle between the multiple states of this button."""
        if self.isChecked():
            newstate = self._onstate
        else:
            newstate = self._offstate
        setattr(self._target, self._attribute, newstate)

    def _on_change(self, event=None):
        """Called wen mirrored value changes

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        with self._events.blocker():
            if self.isChecked() != (
                getattr(self._target, self._attribute) == self._onstate
            ):
                self.toggle()
