from qtpy.QtCore import QModelIndex, QRect
from qtpy.QtGui import QIcon, QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.colormaps import (
    AVAILABLE_COLORMAPS,
    display_name_to_name,
    ensure_colormap,
    make_colorbar,
)
from napari.utils.translations import trans

COLORMAP_WIDTH = 50
TEXT_WIDTH = 130
ENTRY_HEIGHT = 20
PADDING = 1


class ColorStyledDelegate(QStyledItemDelegate):
    """Class for paint :py:class:`~.ColorComboBox` elements when list trigger

    Parameters
    ----------
    base_height : int
        Height of single list element.
    color_dict: dict
        Dict mapping name to colors.
    """

    def __init__(self, base_height: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_height = base_height

    def paint(
        self,
        painter: QPainter,
        style: QStyleOptionViewItem,
        model: QModelIndex,
    ):
        style2 = QStyleOptionViewItem(style)

        cbar_rect = QRect(
            style.rect.x(),
            style.rect.y() + PADDING,
            style.rect.width() - TEXT_WIDTH,
            style.rect.height() - 2 * PADDING,
        )
        text_rect = QRect(
            style.rect.width() - TEXT_WIDTH,
            style.rect.y() + PADDING,
            style.rect.width(),
            style.rect.height() - 2 * PADDING,
        )
        style2.rect = text_rect
        super().paint(painter, style2, model)
        name = display_name_to_name(model.data())
        cbar = make_colorbar(ensure_colormap(name), (18, 100))
        image = QImage(
            cbar,
            cbar.shape[1],
            cbar.shape[0],
            QImage.Format_RGBA8888,
        )
        painter.drawImage(cbar_rect, image)

    def sizeHint(self, style: QStyleOptionViewItem, model: QModelIndex):
        res = super().sizeHint(style, model)
        res.setHeight(self.base_height)
        res.setWidth(max(500, res.width()))
        return res


class QtColormapComboBox(QComboBox):
    """Combobox showing colormaps

    Parameters
    ----------
    parent : QWidget
        Parent widget of comboxbox.
    """

    def __init__(self, parent) -> None:
        super().__init__(parent)
        view = QListView()
        view.setMinimumWidth(COLORMAP_WIDTH + TEXT_WIDTH)
        view.setItemDelegate(ColorStyledDelegate(ENTRY_HEIGHT))
        self.setView(view)


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
    colorbar_label : qtpy.QtWidgets.QPushButton
        Button showing colorbar widget. Also enables selection of custom colormap.
    colormap_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current colormap of the layer.
    colormapWidget : qtpy.QtWidgets.QWidget
        Widget to wrap combobox and label widgets related with the layer colormap attribute.
    colormap_widget_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
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

        comboBox.currentTextChanged.connect(self.change_color)
        self.colormap_combobox = comboBox
        self.colorbar_label = QPushButton(parent=parent)
        self.colorbar_label.setObjectName('colorbar')
        self.colorbar_label.setToolTip(trans._('Colorbar'))
        self.colorbar_label.clicked.connect(self._on_make_colormap)

        colormap_layout = QHBoxLayout()
        colormap_layout.setContentsMargins(0, 0, 0, 2)
        if hasattr(self._layer, 'rgb') and self._layer.rgb:
            colormap_layout.addWidget(QLabel('RGB'))
            self.colormap_combobox.setVisible(False)
            self.colorbar_label.setVisible(False)
        else:
            colormap_layout.addWidget(self.colorbar_label)
            colormap_layout.addWidget(self.colormap_combobox, stretch=1)
        colormap_layout.addStretch(1)
        self.colormapWidget = QWidget()
        self.colormapWidget.setProperty('emphasized', True)
        self.colormapWidget.setLayout(colormap_layout)

        self._on_colormap_change()

        self.colormap_widget_label = QtWrappedLabel(trans._('colormap:'))

    def change_color(self, text):
        """Change colormap on the layer model.

        Parameters
        ----------
        text : str
            Colormap name.
        """
        self._layer.colormap = self.colormap_combobox.currentData()

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update dropdown menu."""
        name = self._layer.colormap.name
        if name not in self.colormap_combobox._allitems and (
            cm := AVAILABLE_COLORMAPS.get(name)
        ):
            self.colormap_combobox._allitems.add(name)
            self.colormap_combobox.addItem(cm._display_name, name)

        if name != self.colormap_combobox.currentData():
            index = self.colormap_combobox.findData(name)
            self.colormap_combobox.setCurrentIndex(index)

        # Note that QImage expects the image width followed by height
        cbar = self._layer.colormap.colorbar
        image = QImage(
            cbar,
            cbar.shape[1],
            cbar.shape[0],
            QImage.Format_RGBA8888,
        )
        self.colorbar_label.setIcon(QIcon(QPixmap.fromImage(image)))

    def _on_make_colormap(self):
        """Make new colormap when colorbar_label (pushbutton) is pressed."""
        from napari._qt.utils import get_color
        from napari.utils.colormaps.colormap_utils import ensure_colormap

        color = get_color(self.parent(), mode='hex')
        if color:
            self._layer.colormap = ensure_colormap(color)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.colormap_widget_label, self.colormapWidget)]
