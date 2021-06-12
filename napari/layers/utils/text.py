import warnings
from typing import Tuple, Union

import numpy as np

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EmitterGroup, Event
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor, TextMode
from ._text_utils import format_text_properties, get_text_anchors


class TextManager:
    """Manages properties related to text displayed in conjunction with the layer.

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
    blending : str
        The blending mode that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}. Note that 'opaque` blending
        is not recommended, as colors the bounding box surrounding the text.
        The default value is 'translucent'
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
    The blending mode that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}. Note that 'opaque` blending
        is not recommended, as colors the bounding box surrounding the text.
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
        color='cyan',
        size=12,
        blending='translucent',
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
            blending=Event,
            visible=Event,
        )

        self.events.block_all()
        self._rotation = rotation
        self._anchor = Anchor(anchor)
        self._translation = translation
        self._color = transform_color(color)[0]
        self._size = size
        self._blending = self._check_blending_mode(blending)
        self._visible = visible

        self._set_text(text, n_text, properties)
        self.events.unblock_all()

    @property
    def values(self):
        """np.ndarray: the text values to be displayed"""
        return self._values

    def _set_text(
        self, text: Union[None, str], n_text: int, properties: dict = {}
    ):
        if text is None:
            text = np.empty(0)
        if n_text == 0 and len(text) != 0:
            # initialize text but don't add text elements
            formatted_text, text_mode = format_text_properties(
                text, n_text, properties
            )
            self._text_format_string = text
            self._mode = text_mode
            self._values = formatted_text
        elif len(properties) == 0 or len(text) == 0:
            # set text mode to NONE if no props/text are provided
            self._mode = TextMode.NONE
            self._text_format_string = ''
            self._values = np.empty(0)
        else:
            formatted_text, text_mode = format_text_properties(
                text, n_text, properties
            )
            self._text_format_string = text
            self._values = formatted_text
            self._mode = text_mode
        self.events.text()

    @property
    def anchor(self) -> str:
        """str: The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right
        '"""
        return str(self._anchor)

    @anchor.setter
    def anchor(self, anchor):
        self._anchor = Anchor(anchor)
        self.events.anchor()

    @property
    def rotation(self) -> float:
        """float: angle of the text elements around the anchor point."""
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        self._rotation = rotation
        self.events.rotation()

    @property
    def translation(self) -> np.ndarray:
        """np.ndarray: offset from the anchor point"""
        return self._translation

    @translation.setter
    def translation(self, translation):
        self._translation = np.asarray(translation)
        self.events.translation()

    @property
    def color(self) -> np.ndarray:
        """np.ndarray: Font color for the text"""
        return self._color

    @color.setter
    def color(self, color):
        self._color = transform_color(color)[0]
        self.events.color()

    @property
    def size(self) -> float:
        """float: Font size of the text."""
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        self.events.size()

    @property
    def blending(self) -> str:
        """Blending mode: Determines how RGB and alpha values get mixed.

        Blending.TRANSLUCENT
            Allows for multiple layers to be blended with different opacity
            and corresponds to depth_test=True, cull_face=False,
            blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
        Blending.ADDITIVE
            Allows for multiple layers to be blended together with
            different colors and opacity. Useful for creating overlays. It
            corresponds to depth_test=False, cull_face=False, blend=True,
            blend_func=('src_alpha', 'one').
        """
        return str(self._blending)

    @blending.setter
    def blending(self, blending):

        self._blending = self._check_blending_mode(blending)
        self.events.blending()

    def _check_blending_mode(self, blending):
        blending_mode = Blending(blending)

        # the opaque blending mode is not allowed for text
        # see: https://github.com/napari/napari/pull/600#issuecomment-554142225
        if blending_mode == Blending.OPAQUE:
            blending_mode = Blending.TRANSLUCENT
            warnings.warn(
                trans._(
                    'opaque blending mode is not allowed for text. setting to translucent.',
                    deferred=True,
                ),
                category=RuntimeWarning,
            )

        return blending_mode

    @property
    def visible(self) -> bool:
        """bool: Set to true of the text should be displayed."""
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible
        self.events.visible()

    @property
    def mode(self) -> str:
        """str: The current text setting mode."""
        return str(self._mode)

    def refresh_text(self, properties: dict):
        """Refresh all of the current text elements using updated properties values

        Parameters
        ----------
        properties : dict
            The new properties from the layer
        """
        self._set_text(
            self._text_format_string,
            n_text=len(self.values),
            properties=properties,
        )

    def add(self, properties: dict, n_text: int):
        """Add a text element using the current format string

        Parameters
        ----------
        properties : dict
            The properties to draw the text from
        n_text : int
            The number of text elements to add
        """
        if self._mode in (TextMode.PROPERTY, TextMode.FORMATTED):
            new_text, _ = format_text_properties(
                self._text_format_string, n_text=n_text, properties=properties
            )

            self._values = np.concatenate((self.values, new_text))

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Remove the indicated text elements

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        if self._mode != TextMode.NONE:
            selected_indices = list(indices_to_remove)
            if len(selected_indices) > 0:
                self._values = np.delete(self.values, selected_indices, axis=0)

    def compute_text_coords(
        self, view_data: np.ndarray, ndisplay: int
    ) -> Tuple[np.ndarray, str, str]:
        """Calculate the coordinates for each text element in view

        Parameters
        ----------
        view_data : np.ndarray
            The in view data from the layer
        ndisplay : int
            The number of dimensions being displayed in the viewer

        Returns
        -------
        text_coords : np.ndarray
            The coordinates of the text elements
        anchor_x : str
            The vispy text anchor for the x axis
        anchor_y : str
            THe vispy text anchor for the y axis
        """
        if len(self.values) > 0:
            anchor_coords, anchor_x, anchor_y = get_text_anchors(
                view_data, ndisplay, self._anchor
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
                text = self.values[indices_view]
            else:
                text = np.array([''])
        else:
            # if no points in this slice send dummy data
            text = np.array([''])

        return text

    def _get_state(self):

        state = {
            'text': self.values,
            'rotation': self.rotation,
            'color': self.color,
            'translation': self.translation,
            'size': self.size,
            'visible': self.visible,
        }

        return state

    def _connect_update_events(
        self, text_update_function, blending_update_function
    ):
        """Function to connect all property update events to the update callback.

        This is typically used in the vispy view file.
        """
        # connect the function for updating the text node
        self.events.text.connect(text_update_function)
        self.events.rotation.connect(text_update_function)
        self.events.translation.connect(text_update_function)
        self.events.anchor.connect(text_update_function)
        self.events.color.connect(text_update_function)
        self.events.size.connect(text_update_function)
        self.events.visible.connect(text_update_function)

        # connect the function for updating the text node blending
        self.events.blending.connect(blending_update_function)

    def __eq__(self, other):
        """Method to test equivalence

        called by: text_manager_1 == text_manager_2
        """
        if isinstance(other, TextManager):
            my_state = self._get_state()
            other_state = other._get_state()
            equal = np.all(
                [
                    np.all(value == other_state[key])
                    for key, value in my_state.items()
                ]
            )

        else:
            equal = False

        return equal

    def __ne__(self, other):
        """Method to test not equal

        called by: text_manager_1 != text_manager_2
        """
        return not (self.__eq__(other))
