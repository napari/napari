from typing import Union

import numpy as np

from ._text_constants import TextMode, Anchor
from .text_utils import (
    format_text_direct,
    format_text_properties,
    get_text_anchors,
)
from ...utils.colormaps.standardize_color import transform_color
from ...utils.event import EmitterGroup, Event


class TextManager:
    """ Manages properties related to text displayed in conjunction with the layer

    Parameters
    ----------
    text : array or str
        the strings to be displayed
    rotation : float
        Angle of the text elements around the anchor point. Default value is 0.
    anchor : str
        The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right'
    translation : np.ndarray
        Offset from the anchor point.
    color : array or str
        Font color for the text
    size : float
        Font size of the text. Default value is 12.
    visible : bool
        Set to true of the text should be displayed.

    Attributes
    ----------
    text : array
        the strings to be displayed
    rotation : float
        Angle of the text elements around the anchor point. Default value is 0.
    anchor : str
        The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right'.
    translation : np.ndarray
        Offset from the anchor point.
    color : array
        Font color for the text
    size : float
        Font size of the text. Default value is 12.
    visible : bool
        Set to true of the text should be displayed.
    """

    def __init__(
        self,
        text,
        n_text,
        properties={},
        rotation=0,
        translation=0,
        anchor='center',
        color='black',
        size=12,
        visible=True,
    ):

        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            text=Event,
            rotation=Event,
            translation=Event,
            anchor=Event,
            color=Event,
            size=Event,
            visible=Event,
        )

        self.events.block_all()
        self.rotation = rotation
        self.anchor = anchor
        self.translation = translation
        self.color = color
        self.size = size
        self.visible = visible

        self._set_text(text, n_text, properties)
        self.events.unblock_all()

    @property
    def text(self):
        """np.ndarray: the text values to be displayed"""
        return self._text

    # @text.setter
    # def text(self, text):
    #
    #     self._text = text

    def _set_text(self, text, n_text: int, properties: dict = {}):
        if len(properties) == 0 or n_text == 0 or text is None:
            self._mode = TextMode.NONE
            self._text_format_string = ''
            self._text = None
        else:
            if isinstance(text, str):
                formatted_text, text_mode = format_text_properties(
                    text, n_text, properties
                )
                if text_mode in (TextMode.PROPERTY, TextMode.FORMATTED):
                    self._text_format_string = text
                else:
                    self._text_format_string = ''
            else:
                formatted_text, text_mode = format_text_direct(text, n_text)
                self._text_format_string = ''
            self._text = formatted_text
            self._mode = text_mode
        self.events.text()

    @property
    def anchor(self):
        """str: The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right
        '"""
        return str(self._anchor)

    @anchor.setter
    def anchor(self, anchor):
        self._anchor = Anchor(anchor)
        self.events.anchor()

    @property
    def rotation(self):
        """float: angle of the text elements around the anchor point."""
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        self._rotation = rotation
        self.events.rotation()

    @property
    def translation(self):
        """np.ndarray: offset from the anchor point"""
        return self._translation

    @translation.setter
    def translation(self, translation):
        self._translation = np.asarray(translation)
        self.events.translation()

    @property
    def color(self):
        """np.ndarray: Font color for the text"""
        return self._color

    @color.setter
    def color(self, color):
        self._color = transform_color(color)[0]
        self.events.color()

    @property
    def size(self):
        """float: Font size of the text."""
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        self.events.size()

    @property
    def visible(self):
        """bool: Set to true of the text should be displayed."""
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible
        self.events.visible()

    @property
    def mode(self):
        return str(self._mode)

    @mode.setter
    def mode(self, mode):
        self._mode = TextMode(mode)

    def add(self, properties, n_text):
        if self._mode in (TextMode.PROPERTY, TextMode.FORMATTED):
            new_text, _ = format_text_properties(
                self._text_format_string, n_text=n_text, properties=properties
            )
        elif self._mode == TextMode.NONE:
            new_text = np.repeat([''], n_text)

        self._text = np.concatenate((self.text, new_text))

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Remove the selected text elements"""
        if self._mode != TextMode.NONE:
            selected_indices = list(indices_to_remove)
            if len(selected_indices) > 0:
                self._text = np.delete(self.text, selected_indices, axis=0)

    def compute_text_coords(self, view_data, ndisplay):
        if self._mode in [TextMode.FORMATTED, TextMode.PROPERTY]:
            anchor_coords, anchor_x, anchor_y = get_text_anchors(
                view_data, self._anchor
            )
            text_coords = anchor_coords + self.translation
        else:
            text_coords = np.zeros((0, ndisplay))
            anchor_x = 'center'
            anchor_y = 'center'
        return text_coords, anchor_x, anchor_y

    def view_text(self, indices_view: np.ndarray) -> np.ndarray:
        """Get the values of the text elements in view

        Parameters
        ----------
        indices_view : (N x 1) np.ndarray
            Indices of the text elements in view

        Returns
       -------
       text : (N x 1) np.ndarray
           Array of text strings for the N text elements in view
        """
        if len(indices_view) > 0:
            if self._mode in [TextMode.FORMATTED, TextMode.PROPERTY]:
                text = self.text[indices_view]
            else:
                text = np.array([''])
        else:
            # if no points in this slice send dummy data
            text = np.array([''])

        return text

    def _get_state(self):

        state = {
            'text': self.text,
            'rotation': self.rotation,
            'color': self.color,
            'translation': self.translation,
            'size': self.size,
            'visible': self.visible,
        }

        return state

    def _connect_update_events(self, update_function):
        """Function to connect all property update events to the update callback.
        This is typically used in the vispy view file.
        """
        self.events.rotation.connect(update_function)
        self.events.translation.connect(update_function)
        self.events.anchor.connect(update_function)
        self.events.color.connect(update_function)
        self.events.size.connect(update_function)
        self.events.visible.connect(update_function)
