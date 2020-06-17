from ...utils.colormaps.standardize_color import transform_color


class TextManager:
    """ Manages properties related to text displayed in conjunction with the layer

    Parameters
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
        rotation=0,
        color='black',
        size=12,
        font='OpenSans',
        visible=True,
    ):

        self._rotation = rotation
        self._color = color
        self._size = size
        self._font = font
        self._visible = visible

        self.text = text

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):

        self._text = text

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        self._rotation = rotation

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
        return self._font

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible

    def _view_text(self, selected_data):
        selected_data = list(selected_data)

        return self.text[selected_data]
