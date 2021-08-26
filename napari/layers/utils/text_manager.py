import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple, Union

import numpy as np
from pydantic import PositiveInt, validator

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor
from ._text_utils import get_text_anchors
from .property_map import (
    ConstantPropertyMap,
    NamedPropertyMap,
    PropertyMapStore,
    TextFormatPropertyMap,
)


class TextManager(EventedModel):
    """Manages properties related to text displayed in conjunction with the layer.

    Attributes
    ----------
    values : np.ndarray
        The text values to be displayed (read-only).
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
    mapping : Callable[[Dict[str, Any]], str]
        A mapping from layer property table row to a text value.
    """

    visible: bool = True
    size: PositiveInt = 12
    color: Array[float, (4,)] = 'cyan'
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0
    mapping: Callable[[Dict[str, Any]], str] = ConstantPropertyMap(constant='')
    _mapping_store: PropertyMapStore

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.mapping.connect(self._on_mapping_changed)
        self._on_mapping_changed()

    @property
    def values(self):
        return np.array(self._mapping_store.values, dtype=str)

    def refresh_text(self, properties: Dict[str, Array]):
        """Refresh all text values from the given layer properties.

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        """
        self._mapping_store.refresh(properties)

    def add(self, properties: Dict[str, Array], num_to_add: int):
        """Adds a number of a new text values based on the given layer properties.

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        num_to_add : int
            The number of text values to add.
        """
        self._mapping_store.add(properties, num_to_add)

    def remove(self, indices: Iterable[int]):
        """Removes some text values by index.

        Parameters
        ----------
        indices : Iterable[int]
            The indices to remove.
        """
        self._mapping_store.remove(indices)

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
        if len(indices_view) > 0:
            return self.values[indices_view]
        # if no points in this slice send dummy data
        return np.array([''])

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

    def _connect_update_events(
        self, text_update_function, blending_update_function
    ):
        """Function to connect all property update events to the update callback.

        This is typically used in the vispy view file.
        """
        # connect the function for updating the text node
        self._mapping_store.events.values.connect(text_update_function)
        self.events.mapping.connect(text_update_function)
        self.events.rotation.connect(text_update_function)
        self.events.translation.connect(text_update_function)
        self.events.anchor.connect(text_update_function)
        self.events.color.connect(text_update_function)
        self.events.size.connect(text_update_function)
        self.events.visible.connect(text_update_function)

        # connect the function for updating the text node blending
        self.events.blending.connect(blending_update_function)

    def _on_mapping_changed(self):
        self._mapping_store = PropertyMapStore(mapping=self.mapping)

    @staticmethod
    def _mapping_from_text(text: str, properties: Dict[str, Array]):
        if text in properties:
            return NamedPropertyMap(name=text)
        elif ('{' in text) and ('}' in text):
            return TextFormatPropertyMap(format_string=text)
        return ConstantPropertyMap(constant=text)

    @classmethod
    def from_layer_kwargs(
        cls,
        text: Union['TextManager', dict, str, Sequence[str], None],
        properties: Dict[str, Array],
        **kwargs,
    ):
        """Create a TextManager from layer keyword arguments and TextManager attributes.

        Parameters
        ----------
        text : Union[TextManager, dict, str, Sequence[str], None]
            The strings to be displayed, or a format string to be filled out using properties.
        properties: Dict[str, Array]
            Stores properties data that will be used to generate strings.
        **kwargs
            The other accepted keyword arguments as named and described as TextManager's attributes.
        """
        if isinstance(text, TextManager):
            manager = text
        else:
            if isinstance(text, dict):
                kwargs = deepcopy(text)
                if 'text' in kwargs:
                    kwargs['mapping'] = cls._mapping_from_text(
                        kwargs['text'], properties
                    )
                    kwargs.pop('text')
            elif isinstance(text, str):
                kwargs['mapping'] = cls._mapping_from_text(text, properties)
            elif isinstance(text, (list, np.ndarray, tuple)):
                # This is direct mode where we add text as a column in the property table.
                properties['_text'] = np.array(text)
                kwargs['mapping'] = NamedPropertyMap(name='_text')
            elif text is None:
                kwargs['mapping'] = ConstantPropertyMap(constant='')
            else:
                raise TypeError(
                    trans._(
                        'text should be a string, array, dict, or TextManager',
                        deferred=True,
                    )
                )
            manager = cls(**kwargs)
        manager.refresh_text(properties)
        return manager
