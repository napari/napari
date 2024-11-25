from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from napari._qt.layer_controls.qt_colormap_combobox import QtColormapComboBox
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.translations import trans


# TODO: Better reusage of code between classes here?
class QtSimpleColormapComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer colormaps
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
        colormap_combobox : qtpy.QtWidgets.QComboBox
            ComboBox controlling current colormap of the layer.
        colormap_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
            Label for the colormap chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.colormap.connect(self._on_colormap_change)

        # Setup widgets
        self.colormap_combobox = QComboBox()
        for name, colormap in AVAILABLE_COLORMAPS.items():
            display_name = colormap._display_name
            self.colormap_combobox.addItem(display_name, name)
        self.colormap_combobox.currentTextChanged.connect(self.change_colormap)

        self.colormap_combobox_label = QtWrappedLabel(trans._('colormap:'))

        self._on_colormap_change()

    def change_colormap(self, colormap: str):
        self._layer.colormap = self.colormap_combobox.currentData()

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update combobox."""
        with self._layer.events.colormap.blocker():
            self.colormap_combobox.setCurrentIndex(
                self.colormap_combobox.findData(self._layer.colormap)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.colormap_combobox_label, self.colormap_combobox)]


class QtColormapControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer colormaps
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
        colorbarLabel : qtpy.QtWidgets.QLabel
            Label text of colorbar widget.
        colormapComboBox : qtpy.QtWidgets.QComboBox
            ComboBox controlling current colormap of the layer.
        colormapWidget : qtpy.QtWidgets.QWidget
            Widget to wrap combobox and label widgets related with the layer colormap attribute.
        colormapWidgetLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
            Label for the color mode chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.colormap.connect(self._on_colormap_change)

        # Setup widgets
        comboBox = QtColormapComboBox(parent)
        comboBox.setObjectName('colormapComboBox')
        comboBox._allitems = set(self._layer.colormaps)

        for name, cm in AVAILABLE_COLORMAPS.items():
            if name in self._layer.colormaps:
                comboBox.addItem(cm._display_name, name)

        comboBox.currentTextChanged.connect(self.changeColor)
        self.colormapComboBox = comboBox
        self.colorbarLabel = QLabel(parent=parent)
        self.colorbarLabel.setObjectName('colorbar')
        self.colorbarLabel.setToolTip(trans._('Colorbar'))

        colormap_layout = QHBoxLayout()
        if hasattr(self._layer, 'rgb') and self._layer.rgb:
            colormap_layout.addWidget(QLabel('RGB'))
            self.colormapComboBox.setVisible(False)
            self.colorbarLabel.setVisible(False)
        else:
            colormap_layout.addWidget(self.colorbarLabel)
            colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)
        self.colormapWidget = QWidget()
        self.colormapWidget.setProperty('emphasized', True)
        self.colormapWidget.setLayout(colormap_layout)

        self._on_colormap_change()

        self.colormapWidgetLabel = QtWrappedLabel(trans._('colormap:'))

    def changeColor(self, text):
        """Change colormap on the layer model.

        Parameters
        ----------
        text : str
            Colormap name.
        """
        self._layer.colormap = self.colormapComboBox.currentData()

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update dropdown menu."""
        name = self._layer.colormap.name
        if name not in self.colormapComboBox._allitems and (
            cm := AVAILABLE_COLORMAPS.get(name)
        ):
            self.colormapComboBox._allitems.add(name)
            self.colormapComboBox.addItem(cm._display_name, name)

        if name != self.colormapComboBox.currentData():
            index = self.colormapComboBox.findData(name)
            self.colormapComboBox.setCurrentIndex(index)

        # Note that QImage expects the image width followed by height
        cbar = self._layer.colormap.colorbar
        image = QImage(
            cbar,
            cbar.shape[1],
            cbar.shape[0],
            QImage.Format_RGBA8888,
        )
        self.colorbarLabel.setPixmap(QPixmap.fromImage(image))

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.colormapWidgetLabel, self.colormapWidget)]
