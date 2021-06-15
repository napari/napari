import warnings
from typing import Optional, Tuple, Union

import numpy as np
from pydantic import validator

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor, TextMode
from ._text_utils import format_text_properties, get_text_anchors


class TextManager(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Parameters
    ----------
    text : array or str
        The strings to be displayed, or a format string that will be filled out
        n_text times using data in properties.
    n_text : int
        The number of the strings to be displayed. This may be different to the
        if text is a format string, or if text should be repeated.
    properties: dict
        Stores properties data that will be used to generate strings when text
        is a format a string.

    Attributes
    ----------
    visible : bool
        True if the text should be displayed, false otherwise.
    size : float
        Font size of the text. Default value is 12.
    color : array
        Font color for the text.
    blending : Blending
        The blending mode that determines how RGB and alpha values of the layer
        visual get mixed. Allowed values are {'opaque', 'translucent', and 'additive'}.
        Note that 'opaque` blending is not recommended, as it colors the bounding box
        surrounding the text.
    anchor : Anchor
        The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right'.
    translation : np.ndarray
        Offset from the anchor point.
    rotation : float
        Angle of the text elements around the anchor point. Default value is 0.
    """

    visible: bool = True
    size: int = 12
    color: np.ndarray = None
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    translation: np.ndarray = None
    rotation: float = 0
    _values: np.ndarray
    _mode: TextMode
    _text_format_string: str

    def __init__(self, text, n_text, properties=None, **kwargs):
        super().__init__(**kwargs)
        self._set_text(text, n_text, properties=properties)

    def _set_text(
        self,
        text: Optional[str],
        n_text: int,
        properties: Optional[dict] = None,
    ):
        if properties is None:
            properties = {}
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

    def refresh_text(self, properties: dict):
        """Refresh all of the current text elements using updated properties values

        Parameters
        ----------
        properties : dict
            The new properties from the layer
        """
        self._set_text(
            self._text_format_string,
            n_text=len(self._values),
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
            self._values = np.concatenate((self._values, new_text))

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
                self._values = np.delete(
                    self._values, selected_indices, axis=0
                )

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
            The vispy text anchor for the y axis
        """
        if len(self._values) > 0:
            anchor_coords, anchor_x, anchor_y = get_text_anchors(
                view_data, ndisplay, self.anchor
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
        if len(indices_view) > 0 and self._mode in [
            TextMode.FORMATTED,
            TextMode.PROPERTY,
        ]:
            return self._values[indices_view]
        # if no points in this slice send dummy data
        return np.array([''])

    @property
    def values(self) -> np.ndarray:
        """np.ndarray: the text values to be displayed"""
        return self._values

    @validator('size')
    def _check_size(cls, size):
        if size <= 0:
            raise ValueError('size must be positive')
        return size

    @validator('color', pre=True, always=True)
    def _check_color(cls, color):
        return transform_color(color or 'cyan')[0]

    @validator('translation', pre=True, always=True)
    def _check_translation(cls, translation):
        # Use a scalar default value of 0 to broadcast to all dimensions.
        return np.asarray(translation or 0)

    @validator('anchor', pre=True, always=True)
    def _check_anchor(cls, anchor):
        return Anchor(anchor)

    @validator('blending', pre=True, always=True)
    def _check_blending_mode(cls, blending):
        blending_mode = Blending(blending)

        # The opaque blending mode is not allowed for text.
        # See: https://github.com/napari/napari/pull/600#issuecomment-554142225
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

    def _connect_update_events(
        self, text_update_function, blending_update_function
    ):
        """Function to connect all property update events to the update callback.
        # the opaque blending mode is not allowed for text
        # see: https://github.com/napari/napari/pull/600#issuecomment-554142225

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
