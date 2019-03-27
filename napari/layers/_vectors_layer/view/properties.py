
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QGridLayout
import numpy as np
import scipy.signal as signal

from ..._base_layer import QtLayer


class QtVectorsLayer(QtLayer):

    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.emit_avg.connect(self._default_avg)
        self.layer.events.emit_len.connect(self._default_length)

        # vector color adjustment and widget
        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        index = face_comboBox.findText(self.layer.color, Qt.MatchFixedString)
        if index >= 0:
            face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.change_face_color(text))
        self.grid_layout.addWidget(QLabel('color:'), 3, 0)
        self.grid_layout.addWidget(face_comboBox, 3, 1)

        # line width in pixels
        width_field = QSpinBox()
        value = self.layer.width
        width_field.setValue(value)
        width_field.valueChanged.connect(self.change_width)
        self.grid_layout.addWidget(QLabel('width:'), 4, 0)
        self.grid_layout.addWidget(width_field, 4, 1)

        # averaging spinbox
        self.averaging_spinbox = QSpinBox()
        self.averaging_spinbox.setSingleStep(2)
        self.averaging_spinbox.setValue(1)
        self.averaging_spinbox.valueChanged.connect(self.change_average_type)
        self.grid_layout.addWidget(QLabel('avg kernel'), 5, 0)
        self.grid_layout.addWidget(self.averaging_spinbox, 5, 1)

        # line length
        self.length_field = QDoubleSpinBox()
        self.length_field.setSingleStep(0.1)
        value = self.layer.length
        self.length_field.setValue(value)
        self.length_field.valueChanged.connect(self.change_length)
        self.grid_layout.addWidget(QLabel('length:'), 6, 0)
        self.grid_layout.addWidget(self.length_field, 6, 1)

        self.setExpanded(False)

    def change_face_color(self, text):
        self.layer.color = text

    def change_connector_type(self, text):
        self.layer.connector = text

    def change_average_type(self, value):
        self.layer.averaging = value

    def change_width(self, value):
        self.layer.width = value
    
    def change_length(self, value):
        self.layer.length = value

    def _default_avg(self, event_kernel):
        """
        Default method for calculating average
        Implemented ONLY for image-like vector data
        :param event_kernel: kernel over which to compute average
        :return:
        """
        if self.layer._data_type == 'coords':
            # default averaging is supported only for 'matrix' dataTypes
            return None
        elif self.layer._data_type == 'image':
            # x = int(event_kernel.name.split('x')[0])
            # y = int(event_kernel.name.split('x')[1])

            x = self.averaging_spinbox.value()
            y = self.averaging_spinbox.value()

            if (x,y) == (1, 1):
                self.layer.vectors = self.layer._original_data
                return None

            tempdat = self.layer._original_data
            range_x = tempdat.shape[0]
            range_y = tempdat.shape[1]
            x_offset = int((x - 1) / 2)
            y_offset = int((y - 1) / 2)

            kernel = np.ones(shape=(x, y)) / (x*y)

            output_mat = np.zeros_like(tempdat)
            output_mat_x = signal.convolve2d(tempdat[:, :, 0], kernel, mode='same', boundary='wrap')
            output_mat_y = signal.convolve2d(tempdat[:, :, 1], kernel, mode='same', boundary='wrap')

            output_mat[:, :, 0] = output_mat_x
            output_mat[:, :, 1] = output_mat_y

            self.layer.vectors = output_mat[x_offset:range_x-x_offset:x, y_offset:range_y-y_offset:y]
            self.layer.length = self.layer._length

    def _default_length(self, event_len):
        """
        Default method for calculating vector lengths
        Implemented ONLY for image-like vector data
        :param event_len: new length
        :return:
        """

        if self.layer._data_type == 'coords':
            return None
        elif self.layer._data_type == 'image':
            self.layer._length = self.length_field.value()
            self.layer._vectors = self.layer._convert_to_vector_type(self.layer._current_data)

