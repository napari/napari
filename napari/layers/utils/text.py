from typing import Tuple, Union, Dict, ClassVar, Optional
import numpy as np

from ..base._base_constants import Blending
from ._text_constants import TextMode, Anchor
from ._text_utils import (
    format_text_properties,
    get_text_anchors,
)
from ...utils.colormaps.standardize_color import transform_color
from ...utils.colormaps.vendored.colors import to_hex
from ...utils.dataclass import dataclass
from dataclasses import InitVar
from typing_extensions import Annotated


# https://github.com/napari/napari/issues/1491
def transform_1_color(color):
    return transform_color(color)[0]


def no_opaque(blending):
    val = Blending(blending)
    if val == Blending.OPAQUE:
        import warnings

        warnings.warn(
            'opaque blending mode is not allowed for text. '
            'setting to translucent.'
        )
        return Blending.TRANSLUCENT
    return val


@dataclass(events=True, properties=True)
class TextManager:
    """Manages properties related to text displayed in conjunction with the layer.

    Parameters
    ----------
    text : array or str
        the strings to be displayed
    n_text : int
        The length of data being annotated.
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
    """

    text: InitVar[Union[np.ndarray, str]]
    n_text: InitVar[int]
    properties: InitVar[Dict[str, np.ndarray]] = None

    rotation: float = 0.0
    translation: Annotated[np.ndarray, None, np.asarray] = np.asarray(0)
    anchor: Annotated[Anchor, str, Anchor] = Anchor.CENTER
    color: Annotated[
        Union[np.ndarray, str], to_hex, transform_1_color
    ] = 'cyan'
    size: float = 12.0
    blending: Annotated[Blending, str, no_opaque] = Blending.TRANSLUCENT
    visible: bool = True
    values: Optional[np.ndarray] = None
    mode: TextMode = TextMode.NONE

    # private attributes not in dataclass
    _text_format_string: ClassVar[str] = ''

    def __post_init__(self, text, n_text, properties):
        self.events.add(text=None)
        # called after __init__
        self._set_text(text, n_text, properties or {})

    def _set_text(
        self, text: Union[None, str], n_text: int, properties: dict = {}
    ):
        if len(properties) == 0 or n_text == 0 or text is None:
            self.mode = TextMode.NONE
            self._text_format_string = ''
            self.values = None
        else:
            self.values, self.mode = format_text_properties(
                text, n_text, properties
            )
            self._text_format_string = text
        # `values` and `text` is slightly duplicated.  If this were changed to
        # "self.events.values", then we wouldn't have to add a dedicated event
        # (since self.text isn't actually used except for initialization)
        self.events.text()

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
        if self._mode in [TextMode.FORMATTED, TextMode.PROPERTY]:
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
