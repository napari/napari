import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union

import numpy as np
from pydantic import PositiveInt, validator

from ...utils.events.evented_model import (
    add_to_exclude_kwarg,
    get_repr_args_without,
)

if TYPE_CHECKING:
    from pydantic.typing import ReprArgs

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import Event, EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor
from ._text_utils import get_text_anchors
from .style_encoding import (
    STRING_ENCODINGS,
    ConstantStringEncoding,
    DirectStringEncoding,
    FormatStringEncoding,
    IdentityStringEncoding,
    is_format_string,
    parse_kwargs_as_encoding,
)


class TextManager(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Attributes
    ----------
    properties : Dict[str, np.ndarray]
        The property values, which typically come from a layer.
    text : Union[STRING_ENCODINGS]
        Defines the string for each text element.
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

    # Declare properties as a generic dict so that a copy is not made on validation
    # and we can rely on a layer and this sharing the same instance.
    properties: dict
    text: Union[STRING_ENCODINGS] = ConstantStringEncoding(constant='')
    color: Array[float, (4,)] = 'cyan'
    visible: bool = True
    size: PositiveInt = 12
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.add(text_update=Event)
        # When most of the fields change, listeners typically respond in the
        # same way, so create a super event that they all emit.
        self.events.text.connect(self.events.text_update)
        self.events.color.connect(self.events.text_update)
        self.events.rotation.connect(self.events.text_update)
        self.events.translation.connect(self.events.text_update)
        self.events.anchor.connect(self.events.text_update)
        self.events.size.connect(self.events.text_update)
        self.events.visible.connect(self.events.text_update)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'properties':
            self._on_properties_changed()

    def refresh_text(self, properties: Dict[str, np.ndarray]):
        """Refresh all text elements from the given layer properties.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties of a layer.
        """
        # This shares the same instance of properties as the layer, so when
        # that instance is modified, we won't detect the change. But when
        # layer.properties is reassigned to a new instance, we will.
        # Therefore, manually call _on_properties_changed to ensure those
        # updates always occur exactly once and this always refreshes derived values.
        with self.events.properties.blocker():
            self.properties = properties
        self._on_properties_changed()

    def _paste(self, strings: np.ndarray):
        self.text._append(strings)

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Removes some text elements by index.

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices to remove.
        """
        self.text._delete(list(indices_to_remove))

    def compute_text_coords(
        self, view_data: np.ndarray, ndisplay: int
    ) -> Tuple[np.ndarray, str, str]:
        """Calculate the coordinates for each text element in view.

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
        if len(view_data) > 0:
            anchor_coords, anchor_x, anchor_y = get_text_anchors(
                view_data, ndisplay, self.anchor
            )
            text_coords = anchor_coords + self.translation
        else:
            text_coords = np.zeros((0, ndisplay))
            anchor_x = 'center'
            anchor_y = 'center'
        return text_coords, anchor_x, anchor_y

    def view_text(self, indices_view) -> np.ndarray:
        """Get the values of the text elements in view

        Parameters
        ----------
        indices_view : (N x 1) slice, range, or indices
            Indices of the text elements in view. Must be usable as indices for np.ndarray.

        Returns
        -------
        text : (N x 1) np.ndarray
            Array of text strings for the N text elements in view
        """
        return self.text._get_array(self.properties, indices_view)

    @classmethod
    def _from_layer_kwargs(
        cls,
        *,
        text: Union['TextManager', dict, str, Sequence[str], None],
        properties: Dict[str, np.ndarray],
    ) -> 'TextManager':
        """Create a TextManager from a layer.

        Parameters
        ----------
        text : Union[TextManager, dict, str, Sequence[str], None]
            Another instance of a TextManager, a dict that contains some of its state,
            a string that may be a constant, a property name, or a format string,
            or sequence of strings specified directly.
        properties : Dict[str, np.ndarray]
            The property values, which typically come from a layer.

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
        kwargs['properties'] = properties
        return cls(**kwargs)

    def __repr_args__(self) -> 'ReprArgs':
        return get_repr_args_without(super().__repr_args__(), {'properties'})

    def json(self, **kwargs) -> str:
        add_to_exclude_kwarg(kwargs, {'properties'})
        return super().json(**kwargs)

    def dict(self, **kwargs) -> Dict[str, Any]:
        add_to_exclude_kwarg(kwargs, {'properties'})
        return super().dict(**kwargs)

    @validator('text', pre=True, always=True)
    def _check_text(
        cls,
        text: Union[str, Sequence[str], Union[STRING_ENCODINGS], dict, None],
        values,
    ) -> Union[STRING_ENCODINGS]:
        properties = values['properties']
        if text is None:
            encoding = ConstantStringEncoding(constant='')
        elif isinstance(text, STRING_ENCODINGS):
            encoding = text
        elif isinstance(text, str):
            if text in properties:
                encoding = IdentityStringEncoding(property_name=text)
            elif is_format_string(properties, text):
                encoding = FormatStringEncoding(format_string=text)
            else:
                encoding = ConstantStringEncoding(constant=text)
        elif isinstance(text, dict):
            encoding = parse_kwargs_as_encoding(STRING_ENCODINGS, **text)
        elif isinstance(text, Sequence):
            encoding = DirectStringEncoding(array=text, default='')
        else:
            raise TypeError(
                trans._(
                    'text should be a string, iterable, StringEncoding, dict, or None',
                    deferred=True,
                )
            )
        return encoding

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

    def _on_properties_changed(self, event=None):
        self.text._clear()


def _properties_equal(left, right):
    if left is right:
        return True
    if not (isinstance(left, dict) and isinstance(right, dict)):
        return False
    if left.keys() != right.keys():
        return False
    for key in left:
        if np.any(left[key] != right[key]):
            return False
    return True


TextManager.__eq_operators__['properties'] = _properties_equal
