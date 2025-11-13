from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from psygnal import Signal, SignalGroup

from napari.utils._dtype import normalize_dtype
from napari.utils.colormaps import ensure_colormap
from napari.utils.events import Event
from napari.utils.status_messages import format_float
from napari.utils.translations import trans
from napari.utils.validators import check_list

if TYPE_CHECKING:
    from typing import Any, Literal, Protocol

    import numpy.typing as npt
    from psygnal import SignalInstance

    from napari.components.overlays import ColorBarOverlay
    from napari.utils.colormaps.colormap import Colormap
    from napari.utils.colormaps.colormap_utils import ValidColormapArg
    from napari.utils.events import EmitterGroup, EventedDict

    class IVMSignalGroupProtocol(Protocol):
        """Protocol for IntensityVisualizationMixin signals.

        These signals cannot be declared as `psygnal.Signal` as
        this is simply the descriptor protocol implementation
        of the actual signal. Instead, we want the instance type
        that is created in the descriptor, which is `psygnal.SignalInstance`.

        .. note::
            The protocol is only for type-checking purposes; it should be type
            hinted as a generic so to describe the actual data it transports
            but this is still an open question:
            https://github.com/pyapp-kit/psygnal/pull/304
        """

        contrast_limits: SignalInstance
        contrast_limits_range: SignalInstance
        gamma: SignalInstance
        colormap: SignalInstance


T = TypeVar('T')


class IVMSignalGroup(SignalGroup):
    """IntensityVisualizationMixin signals.

    The actual signals instances created by the descriptor.
    These should be created in the final Layer class that
    uses the IntensityVisualizationMixin.
    """

    contrast_limits = Signal()
    contrast_limits_range = Signal()
    gamma = Signal()
    colormap = Signal()


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

    # TODO: when declaring a protocol like this,
    # mypy WILL complain because it does not understand that
    # both the ScalarFieldBase, IntensityVisualizationMixin and
    # Image classes will have this attribute which has to be extended
    # with the appropriate signals... how to fix this? The nice thing
    # about the current event system is that it relies on dynamic
    # attributes setting, so we don't have to declare them all the time.
    # TODO: a possible solution is to declare these signals as protocol,
    # and then have the final layer class implementing the actual signals?
    signals: IVMSignalGroupProtocol

    # TODO: these *args and **kwargs are useless here,
    # but mypy seems to be complaining if they are not present,
    # so we have to add them to "digest" any extra arguments
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # this super().__init__() call
        # is necessary given the mixin nature of this class
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

    def reset_contrast_limits(
        self, mode: Literal['data', 'slice'] | None = None
    ) -> None:
        """Scale contrast limits to data range"""
        mode = mode or self._auto_contrast_source
        self._contrast_limits = self._calc_data_range(mode)
        self.contrast_limits = list(self._contrast_limits)

    def _calc_data_range(
        self, mode: Literal['data', 'slice'] = 'data'
    ) -> tuple[float, float]:
        raise NotImplementedError

    def reset_contrast_limits_range(
        self, mode: Literal['data', 'slice'] | None = None
    ) -> None:
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
                    n=2,
                )
            )

        self._contrast_limits_msg = (
            format_float(value[0]) + ', ' + format_float(value[1])
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

    # TODO: this kind of typing is confusing,
    # although I understand the reasoining
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
                    n=2,
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
