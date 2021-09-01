import warnings
from copy import deepcopy
from typing import Callable, Dict, Iterable, Tuple, Union

import numpy as np
from pydantic import PositiveInt, validator

from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ...utils.translations import trans
from ..base._base_constants import Blending
from ._text_constants import Anchor
from ._text_utils import get_text_anchors
from .color_transformations import ColorType
from .property_map import (
    ConstantPropertyMap,
    NamedPropertyMap,
    PropertyMap,
    StyleAttribute,
)

DEFAULT_COLOR = 'cyan'


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
    properties : Dict[str, Array]
        The property values, which typically come from a layer.
    color : StyleAttribute
        A style that defines the color for each text element in colors.values
    text : StyleAttribute
        A style that defines the string for each text element in strings.values
    """

    # Declare properties as a generic dict so that a copy is not made on validation
    # and we can rely on a layer and this sharing the same instance.
    properties: dict
    visible: bool = True
    size: PositiveInt = 12
    blending: Blending = Blending.TRANSLUCENT
    anchor: Anchor = Anchor.CENTER
    # Use a scalar default translation to broadcast to any dimensionality.
    translation: Array[float] = 0
    rotation: float = 0
    text: PropertyMap[str] = ''
    color: StyleAttribute = DEFAULT_COLOR

    @property
    def values(self):
        return np.array(self.text.values, dtype=str)

    @property
    def color_values(self):
        values = self.color.values
        return np.empty((0,)) if len(values) == 0 else transform_color(values)

    def refresh_text(self, properties: Dict[str, Array]):
        """Refresh all text values from the given layer properties.

        Parameters
        ----------
        properties : Dict[str, Array]
            The properties of a layer.
        """
        self.properties = properties

    def add(self, num_to_add: int):
        """Adds a number of a new text values based on the given layer properties.

        Parameters
        ----------
        num_to_add : int
            The number of text values to add.
        """
        self.text.add(self.properties, num_to_add)
        self.color.add(self.properties, num_to_add)

    def remove(self, indices: Iterable[int]):
        """Removes some text values by index.

        Parameters
        ----------
        indices : Iterable[int]
            The indices to remove.
        """
        self.text.remove(indices)
        self.color.remove(indices)

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
        # if no elements in this slice send dummy data
        return np.array([''])

    def view_colors(self, indices_view: np.ndarray) -> np.ndarray:
        """Get the colors of the text elements in view

        Parameters
        ----------
        indices_view : (N x 1) np.ndarray
            Indices of the text elements in view
        Returns
        -------
        colors : (N x 4) np.ndarray
            Array of colors for the N text elements in view
        """
        if len(indices_view) > 0:
            return self.color_values[indices_view]
        # if no elements in this slice send dummy data
        return np.array([''])

    @validator('properties', pre=True, always=True)
    def _check_properties(cls, properties, values):
        if 'text' in values:
            values['text'].refresh(properties)
        if 'color' in values:
            values['color'].refresh(properties)
        return properties

    @validator('text', pre=True, always=True)
    def _check_text(
        cls, text: Union[str, Iterable[str], None], values
    ) -> PropertyMap[str]:
        properties = values['properties']
        if isinstance(text, str):
            property_map = cls._mapping_from_text(text, properties)
        elif isinstance(text, Iterable):
            property_map = PropertyMap.from_iterable(text, '')
        elif text is None:
            property_map = PropertyMap.from_constant('')
        else:
            raise TypeError(
                trans._(
                    'text should be a string, iterable, or None', deferred=True
                )
            )
        property_map.refresh(properties)
        return property_map

    @classmethod
    def _mapping_from_text(
        cls, text: str, properties: Dict[str, Array]
    ) -> PropertyMap[str]:
        if text in properties:
            return PropertyMap.from_property(text)
        elif ('{' in text) and ('}' in text):
            return PropertyMap.from_format_string(text)
        return ConstantPropertyMap(constant=text)

    @validator('color', pre=True, always=True)
    def _check_color(
        cls, color: Union[ColorType, Iterable[ColorType], None], values
    ) -> StyleAttribute:
        properties = values['properties']
        if color is None:
            style = StyleAttribute(
                mapping=ConstantPropertyMap(constant=DEFAULT_COLOR)
            )
        elif isinstance(color, str) and color in properties:
            style = StyleAttribute(mapping=NamedPropertyMap(name=color))
        elif isinstance(color, Callable):
            style = StyleAttribute(mapping=color)
        else:
            color_array = transform_color(color)
            n_colors = color_array.shape[0]
            if n_colors > 1:
                style = StyleAttribute(
                    values=list(color), default_value=DEFAULT_COLOR
                )
            else:
                style = StyleAttribute(
                    mapping=ConstantPropertyMap(constant=color)
                )
        style.refresh(properties)
        return style

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
        self.text.events.values.connect(text_update_function)
        self.color.events.values.connect(text_update_function)
        self.events.rotation.connect(text_update_function)
        self.events.translation.connect(text_update_function)
        self.events.anchor.connect(text_update_function)
        self.events.size.connect(text_update_function)
        self.events.visible.connect(text_update_function)

        # connect the function for updating the text node blending
        self.events.blending.connect(blending_update_function)

    @classmethod
    def from_layer_kwargs(
        cls,
        text: Union['TextManager', dict, str, Iterable[str], None],
        properties: Dict[str, Array],
        **kwargs,
    ):
        """Create a TextManager from layer keyword arguments and TextManager attributes.

        Parameters
        ----------
        text : Union[TextManager, dict, str, Iterable[str], None]
            The strings to be displayed, or a format string to be filled out using properties.
        properties: Dict[str, Array]
            Stores properties data that will be used to generate strings.
        **kwargs
            The other accepted keyword arguments as named and described as TextManager's attributes.
        """
        if isinstance(text, TextManager):
            manager = text
            manager.properties = properties
        elif isinstance(text, dict):
            kwargs = deepcopy(text)
            kwargs['properties'] = properties
            manager = cls(**kwargs)
        else:
            kwargs['text'] = text
            kwargs['properties'] = properties
            manager = cls(**kwargs)
        return manager


def _properties_equal(left, right):
    if not (isinstance(left, dict) and isinstance(right, dict)):
        return False
    if left.keys() != right.keys():
        return False
    for key in left:
        if any(left[key] != right[key]):
            return False
    return True


TextManager.__eq_operators__['properties'] = _properties_equal
