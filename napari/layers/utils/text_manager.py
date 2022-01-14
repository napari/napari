import warnings
from copy import deepcopy
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import PositiveInt, validator

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor
from ._text_utils import get_text_anchors
from .layer_utils import _validate_features
from .string_encoding import (
    ConstantStringEncoding,
    DirectStringEncoding,
    ManualStringEncoding,
    StringArray,
    StringEncoding,
    validate_string_encoding,
)


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
    string : StringEncoding
        Defines the string for each text element. See ``validate_string_encoding``
        for accepted inputs.
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

    string: StringEncoding = ConstantStringEncoding(constant='')
    visible: bool = True
    size: PositiveInt = 12
    color: Array[float, (4,)] = 'cyan'
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0

    # Should be the same features as the layer that owns this.
    _features: pd.DataFrame

    def __init__(self, features=None, properties=None, n_text=0, **kwargs):
        if properties is not None:
            # TODO: deprecation warning about using properties.
            features = _validate_features(properties, num_data=n_text)
        elif features is None:
            features = pd.DataFrame()
        self._features = features

        if 'values' in kwargs and 'string' not in kwargs:
            # _warn_about_deprecated_values_field()
            kwargs['string'] = kwargs.pop('values')
        if 'text' in kwargs and 'string' not in kwargs:
            # _warn_about_deprecated_text_parameter()
            text = kwargs.pop('text')
            if isinstance(text, str) and text in features:
                kwargs['string'] = DirectStringEncoding(feature=text)
            else:
                kwargs['string'] = text
        super().__init__(**kwargs)
        self.string(self._features)

    @property
    def values(self):
        # _warn_about_deprecated_values_field()
        return self.string(self._features)

    def __setattr__(self, key, value):
        if key == 'values':
            # _warn_about_deprecated_values_field()
            self.string = value
        else:
            super().__setattr__(key, value)

    def refresh(self, features: pd.DataFrame):
        """Refresh all of the current text elements using a new features table.

        Parameters
        ----------
        features : pd.DataFrame
            The new features table from the layer.
        """
        self._features = features
        self.string._clear()
        self.events.string()

    def refresh_text(self, properties: Dict[str, np.ndarray]):
        """Refresh all of the current text elements using updated properties values

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The new properties from the layer
        """
        # TODO: warn about deprecated
        features = _validate_features(properties)
        self.refresh(features)

    def add(self, properties: dict, n_text: int):
        """Adds a number of a new text elements.

        Parameters
        ----------
        properties : dict
            The properties to draw the text from
        n_text : int
            The number of text elements to add
        """
        # warnings.warn(
        #     trans._(
        #         'TextManager.add is deprecated. '
        #         'Call TextManager.string instead.'
        #     ),
        #     DeprecationWarning,
        # )
        if isinstance(
            self.string, (ConstantStringEncoding, ManualStringEncoding)
        ):
            return
        new_properties = {
            name: np.repeat(value, n_text, axis=0)
            for name, value in properties.items()
        }
        new_values = self.string._apply(new_properties, range(n_text))
        self.string._append(new_values)

    def _paste(self, strings: StringArray):
        self.string._append(strings)

    def remove(self, indices_to_remove: Union[range, set, list, np.ndarray]):
        """Remove the indicated text elements

        Parameters
        ----------
        indices_to_remove : set, list, np.ndarray
            The indices of the text elements to remove.
        """
        if isinstance(indices_to_remove, set):
            indices_to_remove = list(indices_to_remove)
        self.string._delete(indices_to_remove)

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
        # warnings.warn(
        #    trans._(
        #        'TextManager.view_text is deprecated. '
        #        'Call TextManager.string instead.'
        #    ),
        #    DeprecationWarning,
        # )
        if len(indices_view) == 0 or isinstance(
            self.string, (ConstantStringEncoding, ManualStringEncoding)
        ):
            return np.array([''])
        return self.string(self._features, indices=indices_view)

    @validator('string', pre=True, always=True)
    def _check_string(
        cls,
        string: Union[StringEncoding, dict, str, Sequence[str], None],
    ) -> StringEncoding:
        return validate_string_encoding(string)

    @classmethod
    def _from_layer(
        cls,
        *,
        text: Union['TextManager', dict, str, Sequence[str], None],
        features: pd.DataFrame,
    ) -> 'TextManager':
        """Create a TextManager from a layer.

        Parameters
        ----------
        text : Union[TextManager, dict, str, Sequence[str], None]
            An instance of TextManager, a dict that contains some of its state,
            a string that may be a format string, a constant string, or sequence
            of strings specified directly.
        features : pd.DataFrame
            The features table of a layer.

        Returns
        -------
        TextManager
        """
        if isinstance(text, TextManager):
            kwargs = text.dict()
        elif isinstance(text, dict):
            kwargs = deepcopy(text)
        else:
            # TODO: add deprecation warning about this behavior.
            if isinstance(text, str) and text in features:
                kwargs = {'string': DirectStringEncoding(feature=text)}
            else:
                kwargs = {'string': text}
        kwargs['features'] = features
        return cls(**kwargs)

    def _update_from_layer(
        self,
        *,
        text: Union['TextManager', dict, str, None],
        features: pd.DataFrame,
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
        new_manager = TextManager._from_layer(text=text, features=features)

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


def _warn_about_deprecated_values_field():
    warnings.warn(
        trans._(
            'TextManager.values is deprecated. '
            'Call TextManager.string instead.'
        ),
        DeprecationWarning,
    )


def _warn_about_deprecated_text_parameter():
    warnings.warn(
        trans._('text is a deprecated. Use string instead.'),
        DeprecationWarning,
    )
