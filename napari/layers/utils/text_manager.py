import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

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
    _infer_n_rows,
    is_format_string,
    parse_kwargs_as_encoding,
)


class TextManager(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Attributes
    ----------
    properties : Dict[str, np.ndarray]
        The property values, which typically come from a layer.
    string : Union[STRING_ENCODINGS]
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
    string: Union[STRING_ENCODINGS] = ConstantStringEncoding(constant='')
    color: Array[float, (4,)] = 'cyan'
    visible: bool = True
    size: PositiveInt = 12
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0

    def __init__(self, **kwargs):
        if 'values' in kwargs and 'string' not in kwargs:
            _warn_about_deprecated_values_field()
            kwargs['string'] = kwargs.pop('values')
        if 'text' in kwargs and 'string' not in kwargs:
            _warn_about_deprecated_text_parameter()
            kwargs['string'] = kwargs.pop('text')
        super().__init__(**kwargs)
        self.events.add(text_update=Event)
        # When most of the fields change, listeners typically respond in the
        # same way, so create a super event that they all emit.
        self.events.string.connect(self.events.text_update)
        self.events.color.connect(self.events.text_update)
        self.events.rotation.connect(self.events.text_update)
        self.events.translation.connect(self.events.text_update)
        self.events.anchor.connect(self.events.text_update)
        self.events.size.connect(self.events.text_update)
        self.events.visible.connect(self.events.text_update)

    @property
    def values(self):
        _warn_about_deprecated_values_field()
        n_text = _infer_n_rows(self.string, self.properties)
        return self.string._get_array(self.properties, n_text)

    def __setattr__(self, key, value):
        if key == 'values':
            _warn_about_deprecated_values_field()
            self.string = value
        else:
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
        self.properties = properties

    def add(self, properties: dict, num_to_add: int):
        """Adds a number of a new text elements.

        Parameters
        ----------
        properties : dict
            The current properties to draw the text from.
        num_to_add : int
            The number of text elements to add.
        """
        warnings.warn(
            trans._(
                'TextManager.add is a deprecated method. '
                'Use TextManager.string._array(...) to get the strings instead.'
            ),
            DeprecationWarning,
        )
        # Assumes that the current properties passed have already been appended
        # to the properties table, then calls _get_array to append new values now.
        n_text = _infer_n_rows(self.string, self.properties)
        self.string._get_array(self.properties, n_text)

    def _paste(self, strings: np.ndarray):
        self.string._append(strings)

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Removes some text elements by index.

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices to remove.
        """
        self.string._delete(list(indices_to_remove))

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

    def view_text(self, indices_view: Optional = None) -> np.ndarray:
        """Get the values of the text elements in view

        Parameters
        ----------
        indices_view : (N x 1) slice, range, or indices
            Indices of the text elements in view. If None, all values are returned.
            If not None, must be usable as indices for np.ndarray.

        Returns
        -------
        text : (N x 1) np.ndarray
            Array of text strings for the N text elements in view
        """
        warnings.warn(
            trans._(
                'TextManager.view_text() is a deprecated method. '
                'Use TextManager.string._array(...) to get the strings instead.'
            ),
            DeprecationWarning,
        )
        n_text = _infer_n_rows(self.string, self.properties)
        return self.string._get_array(self.properties, n_text, indices_view)

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
            An instance of TextManager, a dict that contains some of its state,
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
            kwargs = {'string': text}
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

    @validator('string', pre=True, always=True)
    def _check_string(
        cls,
        string: Union[str, Sequence[str], Union[STRING_ENCODINGS], dict, None],
        values,
    ) -> Union[STRING_ENCODINGS]:
        if string is None:
            return ConstantStringEncoding(constant='')
        if isinstance(string, STRING_ENCODINGS):
            return string
        if isinstance(string, str):
            properties = values['properties']
            if string in properties:
                return IdentityStringEncoding(property_name=string)
            if is_format_string(properties, string):
                return FormatStringEncoding(format_string=string)
            return ConstantStringEncoding(constant=string)
        if isinstance(string, dict):
            return parse_kwargs_as_encoding(STRING_ENCODINGS, **string)
        if isinstance(string, Sequence):
            return DirectStringEncoding(array=string, default='')
        raise TypeError(
            trans._(
                'string should be a StringEncoding, string, dict, sequence, or None',
                deferred=True,
            )
        )

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
        self.string._clear()
        self.events.text_update()


def _warn_about_deprecated_values_field():
    warnings.warn(
        trans._(
            '`TextManager.values` is a deprecated field. Use `TextManager.string` instead.'
        ),
        DeprecationWarning,
    )


def _warn_about_deprecated_text_parameter():
    warnings.warn(
        trans._('`text` is a deprecated parameter. Use `string` instead.'),
        DeprecationWarning,
    )


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
