import numpy as np

from ._text_constants import TextMode
from .text_utils import format_text_direct, format_text_properties
from ...utils.colormaps.standardize_color import transform_color


class TextManager:
    """ Manages properties related to text displayed in conjunction with the layer

    Parameters
    ----------
    text : array or str
        the strings to be displayed
    rotation : float
        Angle of the text elements around the data point. Default value is 0.
    color : array or str
        Font color for the text
    size : float
        Font size of the text. Default value is 12.
    font : str
        Font to use for the text.
    visible : bool
        Set to true of the text should be displayed.

    Attributes
    ----------
    text : array or str
        the strings to be displayed
    rotation : float
        Angle of the text elements around the data point. Default value is 0.
    size : float
        Font size of the text. Default value is 12.
    font : str
        Font to use for the text.
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
        color='black',
        size=12,
        font='OpenSans',
        visible=True,
    ):

        self.rotation = rotation
        self.translation = translation
        self.color = color
        self.size = size
        self.font = font
        self.visible = visible

        self.set_text(text, n_text, properties)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):

        self._text = text

    def set_text(self, text, n_text: int, properties: dict = {}):
        if len(properties) == 0:
            formatted_text, text_mode = format_text_direct(text, n_text)
            self._mode = TextMode.DIRECT
            self._text_format_string = ''
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

        self._mode = text_mode
        self.text = formatted_text

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        self._rotation = rotation

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):
        self._translation = translation

    @property
    def color(self):

        return self._color

    @color.setter
    def color(self, color):
        self._color = transform_color(color)[0]

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):

        self._size = size

    @property
    def font(self):

        return self._font

    @font.setter
    def font(self, font):
        self._font = font

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible

    @property
    def mode(self):
        return str(self._mode)

    @mode.setter
    def mode(self, mode):
        self._mode = TextMode(mode)

    def add(self, text, n_text):
        if self._mode == TextMode.DIRECT:
            if isinstance(text, str):
                new_text = np.repeat(text, n_text)
            else:
                new_text = text
        elif self._mode in (TextMode.PROPERTY, TextMode.FORMATTED):
            # text_property = text[self._text_format_string]
            # if len(text_property) == 1:
            #     new_text = np.repeat(text, n_text)
            # else:
            #     new_text = text_property
            # elif self._mode == TextMode.FORMATTED:
            new_text, _ = format_text_properties(
                self._text_format_string, n_text=n_text, properties=text
            )

        self._text = np.concatenate([self.text, new_text])

    def _view_text(self, selected_data):
        selected_data = list(selected_data)

        return self.text[selected_data]

    def _get_state(self):

        state = {
            'text': self.text,
            'rotation': self.rotation,
            'color': self.color,
            'translation': self.translation,
            'size': self.size,
            'font': self.font,
            'visible': self.visible,
        }

        return state
