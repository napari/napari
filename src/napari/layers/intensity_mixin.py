from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from napari.utils._dtype import normalize_dtype
from napari.utils.colormaps import ensure_colormap
from napari.utils.events import Event
from napari.utils.status_messages import format_float
from napari.utils.translations import trans
from napari.utils.validators import check_list

if TYPE_CHECKING:
    from typing import Literal, Protocol

    import numpy.typing as npt
    from psygnal import Signal, SignalGroup

    from napari.components.overlays import ColorBarOverlay
    from napari.utils.colormaps.colormap import Colormap
    from napari.utils.colormaps.colormap_utils import ValidColormapArg
    from napari.utils.events import EmitterGroup, EventedDict

    class IVMSignalGroup(SignalGroup, Protocol):
        contrast_limits: Signal
        contrast_limits_range: Signal
        gamma: Signal
        colormap: Signal

T = TypeVar('T')


class IntensityVisualizationMixin:
    """A mixin that adds gamma, colormap, and contrast limits logic to Layers.

    When used, this should come before the Layer in the inheritance, e.g.:

        class Image(IntensityVisualizationMixin, Layer):
            def __init__(self):
                ...

    Note: `contrast_limits_range` is range extent available on the widget,
    and `contrast_limits` is the visible range (the set values on the widget)
    """
    events: EmitterGroup
    _colormap: Colormap
    _overlays: EventedDict
    dtype: npt.DTypeLike
    _colormaps: dict[str, Colormap]
    signals: IVMSignalGroup

    def __init__(self) -> None:

        # TODO: why? this class does
        # not inherit from anything
        super().__init__()

        self.events.add(
            contrast_limits=Event,
            contrast_limits_range=Event,
            gamma=Event,
            colormap=Event,
        )
        self._gamma = 1.0
        self._colormap_name: str = ''
        self._contrast_limits_msg: str = ''
        self._contrast_limits: tuple[float, float] = (-1.0, 1.0)
        self._contrast_limits_range: tuple[float, float] = (-1.0, 1.0)
        self._auto_contrast_source: Literal['data', 'slice'] = 'slice'
        self._keep_auto_contrast = False

        # circular import
        from napari.components.overlays import ColorBarOverlay

        self._overlays.update({'colorbar': ColorBarOverlay()})

    def reset_contrast_limits(self, mode: Literal['data', 'slice'] | None = None) -> None:
        """Scale contrast limits to data range"""
        mode = mode or self._auto_contrast_source
        self._contrast_limits = self._calc_data_range(mode)
        self.contrast_limits = list(self._contrast_limits)

    def _calc_data_range(
        self, mode: Literal['data', 'slice'] = 'data'
    ) -> tuple[float, float]:
        raise NotImplementedError

    def reset_contrast_limits_range(self, mode: Literal['data', 'slice'] | None = None) -> None:
        """Scale contrast limits range to data type if dtype is an integer,
        or use the current maximum data range otherwise.
        """
        dtype = normalize_dtype(self.dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            self._contrast_limits_range = (info.min, info.max)
            self.contrast_limits_range = list(self._contrast_limits_range)
        else:
            mode = mode if mode is not None else self._auto_contrast_source
            self._contrast_limits_range = self._calc_data_range(mode)
            self.contrast_limits_range = list(self._contrast_limits_range)

    @property
    def colormap(self) -> Colormap:
        """napari.utils.Colormap: colormap for luminance images."""
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: ValidColormapArg) -> None:
        self._set_colormap(colormap)

    def _set_colormap(self, colormap: ValidColormapArg) -> None:
        self._colormap = ensure_colormap(colormap)
        self._update_thumbnail()
        self.events.colormap()
        self.signals.colormap()

    @property
    def colormaps(self) -> tuple[str, ...]:
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps.keys())

    @property
    def colorbar(self) -> ColorBarOverlay:
        """Color Bar overlay."""
        return self._overlays['colorbar']

    @property
    def contrast_limits(self) -> list[float | None]:
        """list of float: Limits to use for the colormap."""
        return list(self._contrast_limits)

    @contrast_limits.setter
    def contrast_limits(self, value: list[float | None]) -> None:
        if not check_list(value, 2):
            raise ValueError(
            trans._(
                'Sequence {sequence} must be of length {n} and contain no None values.',
                deferred=True,
                sequence=value,
                n=2
            )
        )

        self._contrast_limits_msg = (
            format_float(value[0])
            + ', '
            + format_float(value[1])
        )
        self._contrast_limits = (value[0], value[1])
        # make sure range slider is big enough to fit range
        newrange = list(self._contrast_limits_range)
        newrange[0] = min(newrange[0], value[0])
        newrange[1] = max(newrange[1], value[1])
        self._contrast_limits_range = (newrange[0], newrange[1])
        self.contrast_limits_range = list(self._contrast_limits_range)
        self._update_thumbnail()
        self.events.contrast_limits()
        self.signals.contrast_limits()

    @property
    def contrast_limits_range(self) -> list[float | None]:
        """The current valid range of the contrast limits."""
        return list(self._contrast_limits_range)

    @contrast_limits_range.setter
    def contrast_limits_range(self, value: list[float | None]) -> None:
        """Set the valid range of the contrast limits.
        If either value is "None", the current range will be preserved.
        If the range overlaps the current contrast limits, the range will be set
        requested and there will be no change the contrast limits.
        If the requested contrast range limits are completely outside the
        current contrast limits, the range will be set as requested and the
        contrast limits will be reset to the new range.
        """
        if not check_list(value, 2):
            # TODO: a more fine-grained check should
            # be made so that if one element is None,
            # it'll preserve the current range for that element
            raise ValueError(
            trans._(
                'Sequence {sequence} must be of length {n} and contain no None values.',
                deferred=True,
                sequence=value,
                n=2
            )
        )
        if list(value) == self.contrast_limits_range:
            return

        self._contrast_limits_range = (value[0], value[1])
        self.contrast_limits_range = list(self._contrast_limits_range)
        self.events.contrast_limits_range()
        self.signals.contrast_limits_range()

        # make sure that the contrast limits fit within the new range
        # this also serves the purpose of emitting events.contrast_limits()
        # and updating the views/controllers
        if hasattr(self, '_contrast_limits') and any(self._contrast_limits):
            clipped_limits = np.clip(self._contrast_limits, value[0], value[1])
            if clipped_limits[0] < clipped_limits[1]:
                self.contrast_limits = list(clipped_limits)
            else:
                self.contrast_limits = list(value)

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = float(value)
        self._update_thumbnail()
        self.events.gamma()
        self.signals.gamma()

    @abstractmethod
    def _update_thumbnail(self) -> None:
        raise NotImplementedError
