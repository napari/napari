import warnings
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from pydantic import PositiveInt, validator

from ...utils.events.evented_model import (
    add_to_exclude_kwarg,
    get_repr_args_without,
)
from .color_encoding import (
    ColorArray,
    ColorEncoding,
    ConstantColorEncoding,
    validate_color_encoding,
)
from .color_transformations import ColorType

if TYPE_CHECKING:
    from pydantic.typing import ReprArgs

from ...utils.events import Event, EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor
from ._text_utils import get_text_anchors
from .string_encoding import (
    ConstantStringEncoding,
    StringArray,
    StringEncoding,
    validate_string_encoding,
)


class TextManager(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Attributes
    ----------
    properties : Dict[str, np.ndarray]
        The property values, which typically come from a layer.
    string : StringEncoding
        Defines the string for each text element. See ``validate_string_encoding``
        for accepted inputs.
    color : ColorEncoding
        Defines the color for each text element. See ``validate_color_encoding``
        for accepted inputs.
    visible : bool
        True if the text should be displayed, false otherwise.
    size : float
        Font size of the text, which must be positive. Default value is 12.
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
    string: StringEncoding = ConstantStringEncoding(constant='')
    color: ColorEncoding = ConstantColorEncoding(constant='cyan')
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

    @property
    def values(self):
        _warn_about_deprecated_values_field()
        n_text = _infer_n_text(self.string, self.properties)
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
                'Use TextManager.string._get_array(...) instead.'
            ),
            DeprecationWarning,
        )
        # Assumes that the current properties passed have already been appended
        # to the properties table, then calls _get_array to append new values now.
        n_text = _infer_n_text(self.string, self.properties)
        self.string._get_array(self.properties, n_text)
        self.color._get_array(self.properties, n_text)

    def _paste(self, strings: StringArray, colors: ColorArray):
        self.string._append(strings)
        self.color._append(colors)

    def remove(self, indices_to_remove: Union[set, list, np.ndarray]):
        """Removes some text elements by index.

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices to remove.
        """
        self.string._delete(list(indices_to_remove))
        self.color._delete(list(indices_to_remove))

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
                'Use TextManager.string._get_array(...) instead.'
            ),
            DeprecationWarning,
        )
        n_text = _infer_n_text(self.string, self.properties)
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
        string: Union[StringEncoding, dict, str, Iterable[str], None],
        values,
    ) -> StringEncoding:
        return validate_string_encoding(string, values['properties'])

    @classmethod
    def _from_layer(
        cls,
        *,
        text: Union['TextManager', dict, str, None],
        properties: Dict[str, np.ndarray],
    ) -> 'TextManager':
        """Create a TextManager from a layer.

        Parameters
        ----------
        text : Union[TextManager, dict, str, None]
            An instance of TextManager, a dict that contains some of its state,
            a string that should be a property name, or a format string.
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
            kwargs = {'string': text}
        kwargs['properties'] = properties
        return cls(**kwargs)

    def _update_from_layer(
        self,
        *,
        text: Union['TextManager', dict, str, None],
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
        new_manager = TextManager._from_layer(text=text, properties=properties)

        # Update a copy of this so that any associated errors are raised
        # before actually making the update. This does not need to be a
        # deep copy because update will only try to reassign fields and
        # should not mutate any existing fields in-place.
        current_manager = self.copy()
        current_manager.update(new_manager, recurse=False)

        # If we got here, then there were no errors, so update for real.
        # Connected callbacks may raise errors, but those are bugs.
        self.update(new_manager, recurse=False)

    @validator('color', pre=True, always=True)
    def _check_color(
        cls,
        color: Union[
            ColorEncoding, dict, ColorType, Iterable[ColorType], None
        ],
        values,
    ) -> ColorEncoding:
        return validate_color_encoding(color, values['properties'])

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
        self.color._clear()
        # TODO: the top-level event is useful because we know that the vispy layer is
        # connected to that, but that might change in the future. Consider emitting the
        # string and color events, which may be less efficient, but is likely safer.
        self.events(Event('_on_properties_changed'))


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


def _infer_n_text(encoding, properties: Dict[str, np.ndarray]) -> int:
    """Infers the number of rows in the given properties table."""
    if len(properties) > 0:
        return len(next(iter(properties)))
    if hasattr(encoding, 'array'):
        return len(encoding.array)
    return 1


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
