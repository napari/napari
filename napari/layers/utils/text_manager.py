import warnings
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import numpy as np
from pydantic import PositiveInt, validator

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor, TextMode
from ._text_utils import format_text_properties, get_text_anchors


class TextManager(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Parameters
    ----------
    text : str
        A a property name or a format string containing property names.
        This will be used to fill out string values n_text times using the
        data in properties.
    n_text : int
        The number of text elements to initially display, which should match
        the number of elements (e.g. points) in a layer.
    properties: dict
        Stores properties data that will be used to generate strings from the
        given text. Typically comes from a layer.

    Attributes
    ----------
    values : np.ndarray
        The text values to be displayed.
    visible : bool
        True if the text should be displayed, false otherwise.
    size : float
        Font size of the text, which must be positive. Default value is 12.
    color : array
        Font color for the text as an [R, G, B, A] array. Can also be expressed
        as a string on construction or setting.
    blending : Blending
        The blending mode that determines how RGB and alpha values of the layer
        visual get mixed. Allowed values are 'translucent' and 'additive'.
        Note that 'opaque` blending is not allowed, as it colors the bounding box
        surrounding the text, and if given, 'translucent' will be used instead.
    anchor : Anchor
        The location of the text origin relative to the bounding box.
        Should be 'center', 'upper_left', 'upper_right', 'lower_left', or 'lower_right'.
    translation : np.ndarray
        Offset from the anchor point.
    rotation : float
        Angle of the text elements around the anchor point. Default value is 0.
    """

    values: Array[str] = []
    visible: bool = True
    size: PositiveInt = 12
    color: Array[float, (4,)] = 'cyan'
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0
    _mode: TextMode = TextMode.NONE
    _text_format_string: str = ''

    def __init__(self, text=None, n_text=None, properties=None, **kwargs):
        super().__init__(**kwargs)
        if 'values' in kwargs:
            text = kwargs['values']
            n_text = len(text)
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
        if len(properties) == 0 or len(text) == 0:
            # set text mode to NONE if no props/text are provided
            self._mode = TextMode.NONE
            self._text_format_string = ''
            self.values = np.empty(0)
        else:
            formatted_text, text_mode = format_text_properties(
                text, n_text, properties
            )
            self._text_format_string = text
            self._mode = text_mode
            self.values = formatted_text

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
            self.values = np.concatenate((self.values, new_text))

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
                self.values = np.delete(self.values, selected_indices, axis=0)

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
        if len(self.values) > 0:
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
            return self.values[indices_view]
        # if no points in this slice send dummy data
        return np.array([''])

    @classmethod
    def _from_layer(
        cls,
        *,
        text: Union['TextManager', dict, str, None],
        n_text: int,
        properties: Dict[str, np.ndarray],
    ) -> 'TextManager':
        """Create a TextManager from a layer.

        Parameters
        ----------
        text : Union[TextManager, dict, str, None]
            An instance of TextManager, a dict that contains some of its state,
            a string that should be a property name, or a format string.
        n_text : int
            The number of text elements to initially display, which should match
            the number of elements (e.g. points) in a layer.
        properties : Dict[str, np.ndarray]
            The properties of a layer.

        Returns
        -------
        TextManager
        """
        if isinstance(text, TextManager):
            kwargs = text.dict()
        elif isinstance(text, dict):
            kwargs = deepcopy(text)
        else:
            kwargs = {'text': text}
        kwargs['n_text'] = n_text
        kwargs['properties'] = properties
        return cls(**kwargs)

    def _update_from_layer(
        self,
        *,
        text: Union['TextManager', dict, str, None],
        n_text: int,
        properties: Dict[str, np.ndarray],
    ):
        """Updates this in-place from a layer.

        This will effectively overwrite all existing state, but in-place
        so that there is no need for any external components to reconnect
        to any useful events. For this reason, only fields that change in
        value will emit their corresponding events.

        Parameters
        ----------
        See :meth:`TextManager._from_layer`.
        """
        # Create a new instance from the input to populate all fields.
        new_manager = TextManager._from_layer(
            text=text, n_text=n_text, properties=properties
        )

        # Update a copy of this so that any associated errors are raised
        # before actually making the update. This does not need to be a
        # deep copy because update will only try to reassign fields and
        # should not mutate any existing fields in-place.
        current_manager = self.copy()
        current_manager.update(new_manager)

        # If we got here, then there were no errors, so update for real.
        # Connected callbacks may raise errors, but those are bugs.
        self.update(new_manager)

        self._mode = new_manager._mode
        self._text_format_string = new_manager._text_format_string

    @validator('color', pre=True, always=True)
    def _check_color(cls, color):
        return transform_color(color)[0]

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
