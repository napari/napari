from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Image
from napari.layers.image._image_constants import VolumeDepiction
from napari.utils.action_manager import action_manager
from napari.utils.translations import trans


class PlaneNormalButtons(QWidget):
    """Qt buttons for controlling plane orientation.

    Attributes
    ----------
    x_button : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the x axis.
    y_button : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the y axis.
    z_button : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the z axis.
    oblique_button : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the camera view direction.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.x_button = QPushButton('x')
        self.y_button = QPushButton('y')
        self.z_button = QPushButton('z')
        self.oblique_button = QPushButton(trans._('oblique'))
        action_manager.bind_button(
            'napari:orient_plane_normal_along_z',
            self.z_button,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_y',
            self.y_button,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_x',
            self.x_button,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_view_direction_no_gen',
            self.oblique_button,
        )

        self.layout().addWidget(self.x_button)
        self.layout().addWidget(self.y_button)
        self.layout().addWidget(self.z_button)
        self.layout().addWidget(self.oblique_button)


class QtDepictionControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer depection
    and plane value attributes and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Image
        An instance of a napari Image layer.

    Attributes
    ----------
    depiction_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current depiction value of the layer.
    depiction_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the depiction value chooser widget.
    plane_normal_buttons : PlaneNormalButtons
        Buttons controlling plane normal orientation when the `plane` depiction value is choosed.
    plane_normal_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal value chooser widget.
    plane_thickness_slider : superqt.QLabeledDoubleSlider
        Slider controlling plane normal thickness when the `plane` depiction value is choosed.
    plane_thickness_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal thickness value chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Image) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.depiction.connect(self._on_depiction_change)
        self._layer.plane.events.thickness.connect(
            self._on_plane_thickness_change
        )

        # Setup widgets
        self.depiction_combobox = QComboBox(parent)
        depiction_options = [d.value for d in VolumeDepiction]
        self.depiction_combobox.addItems(depiction_options)
        index = self.depiction_combobox.findText(
            self._layer.depiction, Qt.MatchFlag.MatchFixedString
        )
        self.depiction_combobox.setCurrentIndex(index)
        self.depiction_combobox.currentTextChanged.connect(
            self.change_depiction
        )
        self.depiction_label = QtWrappedLabel(trans._('depiction:'))

        # plane controls
        self.plane_normal_buttons = PlaneNormalButtons(parent)
        self.plane_normal_label = QtWrappedLabel(trans._('plane normal:'))

        self.plane_thickness_slider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal, parent
        )
        self.plane_thickness_slider.setFocusPolicy(Qt.NoFocus)
        self.plane_thickness_slider.setMinimum(1)
        self.plane_thickness_slider.setMaximum(50)
        self.plane_thickness_slider.setValue(self._layer.plane.thickness)
        self.plane_thickness_slider.valueChanged.connect(
            self.change_plane_thickness
        )
        self.plane_thickness_label = QtWrappedLabel(
            trans._('plane thickness:')
        )

    def change_depiction(self, text: str) -> None:
        self._layer.depiction = text
        self._update_plane_parameter_visibility()

    def change_plane_thickness(self, value: float) -> None:
        self._layer.plane.thickness = value

    def _on_depiction_change(self) -> None:
        """Receive layer model depiction change event and update combobox."""
        with qt_signals_blocked(self.depiction_combobox):
            index = self.depiction_combobox.findText(
                self._layer.depiction, Qt.MatchFlag.MatchFixedString
            )
            self.depiction_combobox.setCurrentIndex(index)
            self._update_plane_parameter_visibility()

    def _on_plane_thickness_change(self) -> None:
        with self._layer.plane.events.blocker():
            self.plane_thickness_slider.setValue(self._layer.plane.thickness)

    def _on_display_change_hide(self) -> None:
        self.depiction_combobox.hide()
        self.depiction_label.hide()

    def _on_display_change_show(self) -> None:
        self.depiction_combobox.show()
        self.depiction_label.show()

    def _update_plane_parameter_visibility(self) -> None:
        """Hide plane rendering controls if they aren't needed."""
        depiction = VolumeDepiction(self._layer.depiction)
        # TODO: Better way to handle the ndisplay value?
        visible = (
            depiction == VolumeDepiction.PLANE
            and self.parent().ndisplay == 3
            and self._layer.ndim >= 3
        )
        self.plane_normal_buttons.setVisible(visible)
        self.plane_normal_label.setVisible(visible)
        self.plane_thickness_slider.setVisible(visible)
        self.plane_thickness_label.setVisible(visible)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.depiction_label, self.depiction_combobox),
            (self.plane_normal_label, self.plane_normal_buttons),
            (self.plane_thickness_label, self.plane_thickness_slider),
        ]
