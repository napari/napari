from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from napari.layers import Layer
from napari.utils._dtype import normalize_dtype
from napari.utils.colormaps import AVAILABLE_COLORMAPS, ensure_colormap
from napari.utils.events import Event
from napari.utils.status_messages import format_float
from napari.utils.validators import _validate_increasing, validate_n_seq

validate_2_tuple = validate_n_seq(2)

if TYPE_CHECKING:
    from napari.layers._scalar_field.scalar_field import ScalarFieldBase
    from napari.layers.surface import Surface


def _validate_contrast_limits(
    values: Sequence[float], name: str
) -> tuple[float, float]:
    """Validate a pair of contrast limits and return them as a tuple of floats.

    Parameters
    ----------
    values : sequence of float
        The two limits to validate. Any 2-length sequence is accepted
        (e.g. a list, a tuple or a numpy array).
    name : str
        Name of the property being validated. Used in error messages.

    Returns
    -------
    tuple of float
        The validated limits.

    Raises
    ------
    TypeError
        If the values are not a pair of numbers.
    ValueError
        If the values are not strictly increasing.
    """
    validate_2_tuple(values)
    # `validate_2_tuple` does not validate generators, so make sure we
    # really do end up with a pair of numbers.
    values = tuple(values)
    if len(values) != 2:
        raise ValueError(f'object must have length 2, got {len(values)}')
    if any(value is None for value in values):
        raise TypeError(
            f'{name} must be a pair of numbers, got {values!r}. '
            'None is not a valid contrast limit.'
        )
    _validate_increasing(values)
    try:
        return float(values[0]), float(values[1])
    except (TypeError, ValueError) as e:
        raise TypeError(
            f'{name} must be a pair of numbers, got {values!r}.'
        ) from e


class IntensityVisualizationMixin(Layer):
    """A mixin that adds gamma, colormap, and contrast limits logic to Layers.

    When used, this should come before the Layer in the inheritance, e.g.:

        class Image(IntensityVisualizationMixin, Layer):
            def __init__(self):
                ...

    Note: `contrast_limits_range` is range extent available on the widget,
    and `contrast_limits` is the visible range (the set values on the widget)
    """

    def __init__(self, *args, **kwargs) -> None:
        # initialize clims before calling super().__init__()
        # so that they are always in a valid state, even if a subclass
        # accesses or sets them while initializing
        self._contrast_limits: tuple[float, float] = (0.0, 1.0)
        self._contrast_limits_range: tuple[float, float] = (0.0, 1.0)

        super().__init__(*args, **kwargs)

        self.events.add(
            contrast_limits=Event,
            contrast_limits_range=Event,
            gamma=Event,
            colormap=Event,
            auto_contrast=Event,
        )
        self._gamma = 1.0
        self._colormap_name = ''
        self._contrast_limits_msg = ''
        self._auto_contrast_source = 'slice'
        self._auto_contrast = False

        # circular import
        from napari.components.overlays import ColorBarOverlay

        self._overlays.update({'colorbar': ColorBarOverlay()})

    def reset_contrast_limits(self: 'ScalarFieldBase | Surface', mode=None):
        """Scale contrast limits to data range"""
        mode = mode or self._auto_contrast_source
        self.contrast_limits = self._calc_data_range(mode)

    def _calc_data_range(self, mode):
        raise NotImplementedError

    def reset_contrast_limits_range(
        self, mode: Literal['data', 'slice'] | None = None
    ) -> None:
        """Scale contrast limits range to data type if dtype is an integer,
        or use the current maximum data range otherwise.

        Parameters
        ----------
        mode: 'data' or 'slice'
           If range should be calculated on whole data, or only visible slice.
        """
        dtype = normalize_dtype(self.dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            self.contrast_limits_range = (info.min, info.max)
        else:
            mode = mode or self._auto_contrast_source
            self.contrast_limits_range = self._calc_data_range(mode)

    @property
    def colormap(self):
        """napari.utils.Colormap: colormap for luminance images."""
        return self._colormap

    @colormap.setter
    def colormap(self, colormap):
        self._set_colormap(colormap)

    def _set_colormap(self, colormap):
        self._colormap = ensure_colormap(colormap)
        self._update_thumbnail()
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(AVAILABLE_COLORMAPS.keys())

    @property
    def colorbar(self):
        """Color Bar overlay."""
        return self._overlays['colorbar']

    @property
    def contrast_limits(self) -> tuple[float, float]:
        """Limits to use for the colormap.

        .. versionchanged:: 0.10.0
            This is now a ``tuple`` of ``float`` instead of a ``list`` that
            could contain ``None``.

        Returns
        -------
        tuple of float
            The ``(min, max)`` intensity values used for the colormap.
        """
        return self._contrast_limits

    @contrast_limits.setter
    def contrast_limits(self, contrast_limits: Sequence[float]) -> None:
        cmin, cmax = _validate_contrast_limits(
            contrast_limits, 'contrast_limits'
        )
        self._contrast_limits_msg = (
            format_float(cmin) + ', ' + format_float(cmax)
        )
        self._contrast_limits = (cmin, cmax)
        # make sure range slider is big enough to fit range
        range_min, range_max = self._contrast_limits_range
        self.contrast_limits_range = (
            min(range_min, cmin),
            max(range_max, cmax),
        )
        self._update_thumbnail()
        self.events.contrast_limits()

    @property
    def contrast_limits_range(self) -> tuple[float, float]:
        """The current valid range of the contrast limits.

        .. versionchanged:: 0.10.0
            This is now a ``tuple`` of ``float`` instead of a ``list`` that
            could contain ``None``.

        Returns
        -------
        tuple of float
            The ``(min, max)`` extent of the allowed contrast limits.
        """
        return self._contrast_limits_range

    @contrast_limits_range.setter
    def contrast_limits_range(self, value: Sequence[float]) -> None:
        """Set the valid range of the contrast limits.

        If the range overlaps the current contrast limits, the range will be
        set as requested and there will be no change to the contrast limits.
        If the requested contrast range limits are completely outside the
        current contrast limits, the range will be set as requested and the
        contrast limits will be reset to the new range.

        .. versionchanged:: 0.10.0
            ``None`` is no longer accepted as a placeholder for "keep the
            current value"; pass an explicit pair of numbers instead.
        """
        new_range = _validate_contrast_limits(value, 'contrast_limits_range')
        if new_range == self._contrast_limits_range:
            return

        self._contrast_limits_range = new_range
        self.events.contrast_limits_range()

        # make sure that the contrast limits fit within the new range
        # this also serves the purpose of emitting events.contrast_limits()
        # and updating the views/controllers
        cmin, cmax = self._contrast_limits
        cmin = max(cmin, new_range[0])
        cmax = min(cmax, new_range[1])
        if cmin >= cmax:
            cmin, cmax = new_range
        self.contrast_limits = (cmin, cmax)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = float(value)
        self._update_thumbnail()
        self.events.gamma()

    @property
    def auto_contrast(self):
        """bool: Whether to automatically set contrast limits to the data range of each slice."""
        return self._auto_contrast

    @auto_contrast.setter
    def auto_contrast(self, value):
        self._auto_contrast = bool(value)
        self.events.auto_contrast()
        if self._auto_contrast:
            self.reset_contrast_limits()
