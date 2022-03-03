from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLayout,
    QWidget,
)

from ...layers.base._base_constants import BLENDING_TRANSLATIONS
from ...utils.events import disconnect_events
from ...utils.translations import trans
from ..widgets._slider_compat import QDoubleSlider


class LayerListGridLayout(QGridLayout):
    """Reusable grid layout for subwidgets in each QtLayerControls class"""

    def __init__(self, QWidget=None):
        super().__init__(QWidget)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(2)
        self.setColumnMinimumWidth(0, 94)
        self.setColumnStretch(1, 1)
        self.setSpacing(4)


class QtLayerControls(QFrame):
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    blendComboBox : qtpy.QtWidgets.QComboBox
        Drowpdown widget to select blending mode of layer.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Layer
        An instance of a napari layer.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    """

    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.opacity.connect(self._on_opacity_change)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.grid_layout = LayerListGridLayout(self)
        self.setLayout(self.grid_layout)

        sld = QDoubleSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(1)
        sld.setSingleStep(0.01)
        sld.valueChanged.connect(self.changeOpacity)
        self.opacitySlider = sld
        self._on_opacity_change()

        blend_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(BLENDING_TRANSLATIONS.items()):
            data = data.value
            blend_comboBox.addItem(text, data)
            if data == self.layer.blending:
                blend_comboBox.setCurrentIndex(index)

        blend_comboBox.activated[str].connect(self.changeBlending)
        self.blendComboBox = blend_comboBox

    def _populate_grid(self, *grid_scheme):
        """
        Populate the layer control grid layout based on a scheme in the form:
        [
            (Qwidget, ..., QWidget),
            (...,),
            (None, QWidget,)
        ]

        Ellipsis means "increase the column span of the previous widget" if part of a row, and
        "increase the row span of all the widgets in the previous row" if part of a column.
        Trailing Ellipses are assumed for rows with fewer entries than others.

        None leaves an empty space in the grid.
        """
        # arguments that will be passed to grid.addWidget
        # list of tuples in the form (widget, row, column, row_span, column_span)
        args = []

        max_cols = max(len(row) for row in grid_scheme)
        previous_row = -1
        for row, widgets in enumerate(grid_scheme):
            if widgets == (None,):
                continue
            if widgets == (Ellipsis,):
                # increment row span of all widgets in the previous row
                for arg in args:
                    if arg[1] == previous_row:
                        arg[3] += 1
                continue
            previous_row = row

            previous_col = -1
            for col, wdg in enumerate(widgets):
                if wdg is None:
                    continue
                if wdg is Ellipsis:
                    # increment column span of the previous widget
                    for arg in args:
                        if arg[1] == row and arg[2] == previous_col:
                            arg[4] += 1
                    continue
                previous_col = col

                # generate simple label with translation if only a string was passed
                if isinstance(wdg, str):
                    wdg = QLabel(trans._(wdg))

                args.append([wdg, row, col, 1, 1])

            # pad if necessary
            empty_space = max_cols - col + 1
            for arg in args:
                if arg[1] == row and arg[2] == previous_col:
                    arg[4] += empty_space

        for arg in args:
            if isinstance(arg[0], QWidget):
                add_item = self.grid_layout.addWidget
            elif isinstance(arg[0], QLayout):
                add_item = self.grid_layout.addLayout
            add_item(*arg)

        self.grid_layout.setRowStretch(previous_row + 1, 1)

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
