"""
This file contains implementation of spatial information for layers.

This information is:

* Scale
* Translates
* Rotate
* Shear
* Unit
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np
import pint
from psygnal import Signal, SignalGroup

from napari.layers.utils.layer_utils import coerce_affine
from napari.utils.transforms import Affine, CompositeAffine, TransformChain

if TYPE_CHECKING:
    import numpy.typing as npt

    UnitsLike = Union[None, str, pint.Unit, dict[str, Union[str, pint.Unit]]]
    UnitsInfo = Union[None, pint.Unit, dict[str, pint.Unit]]


__all__ = ('SpatialInformation',)
_OPTIONAL_PARAMETERS = {
    'affine',
    'axes_labels',
    'rotate',
    'scale',
    'shear',
    'translate',
    'units',
}


class SpatialInformationEvents(SignalGroup):
    affine = Signal(Affine)
    axes_labels = Signal(tuple)
    rotate = Signal(np.ndarray)
    scale = Signal(tuple)
    shear = Signal(np.ndarray)
    translate = Signal(tuple)
    units = Signal(dict)


class SpatialInformation:
    """Spatial information for layers.

    Parameters
    ----------
    ndim: int, optional
        Number of dimensions to be represented.
    affine : array-like or napari.utils.transforms.Affine, optional
        An existing affine transform object or an array-like that is its transform matrix.
    axes_labels : Sequence[str], optional
        Sequence of length `ndim` containing the labels for each axis.
    rotate : float, 3-tuple of float, or n-D array, optional
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float, optional
        Scale factors for the layer. Length of scale should be equal to ndim.
    shear : 1-D array or n-D array, optional
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    translate : tuple of float, optional
        Translation values for the layer. Length of translate should be equal to ndim.
    units :

    scale : tuple of float
        Scale factors for the layer.


    Attributes
    ----------
    affine : napari.utils.transforms.Affine
        Affine transform object.


    """

    def __init__(
        self,
        *,
        ndim: int,
        affine: Union[npt.ArrayLike, Affine, None] = None,
        axes_labels: Optional[Sequence[str]] = None,
        rotate: Union[
            float, Tuple[float, float, float], npt.ArrayLike, None
        ] = None,
        scale: Optional[Sequence[float]] = None,
        shear: Optional[npt.ArrayLike] = None,
        translate: Optional[Sequence[float]] = None,
        units: UnitsLike = None,
    ):
        locals_dict = locals()
        self._parameters_with_default_values = {
            x for x in _OPTIONAL_PARAMETERS if locals_dict[x] is None
        }
        # `self._parameters_with_default_values` is used to store parameters that are not provided
        # during initialization and not set during the lifetime of the object.
        # this will allow checking which attributes have default values and
        # could.should be set when adding a layer to the viewer.
        self.events = SpatialInformationEvents()
        self._ndim = ndim
        self._units, self._axes_labels = _coerce_units_and_axes(
            units, axes_labels
        )

        if scale is None:
            scale = [1] * ndim
        if translate is None:
            translate = [0] * ndim
        self._transforms: TransformChain[Affine] = TransformChain(
            [
                Affine(np.ones(ndim), np.zeros(ndim), name='tile2data'),
                CompositeAffine(
                    scale,
                    translate,
                    rotate=rotate,
                    shear=shear,
                    ndim=ndim,
                    name='data2physical',
                ),
                coerce_affine(affine, ndim=ndim, name='physical2world'),
                Affine(np.ones(ndim), np.zeros(ndim), name='world2grid'),
            ]
        )

    @property
    def affine(self) -> Affine:
        """napari.utils.transforms.Affine: Extra affine transform to go from physical to world coordinates."""
        return self._transforms['physical2world']

    @affine.setter
    def affine(self, affine: Union[npt.ArrayLike, Affine]) -> None:
        # Assignment by transform name is not supported by TransformChain and
        # EventedList, so use the integer index instead. For more details, see:
        # https://github.com/napari/napari/issues/3058
        self._transforms[2] = coerce_affine(
            affine, ndim=self.ndim, name='physical2world'
        )
        self._parameters_with_default_values.discard('affine')
        self.events.affine.emit(self.affine)

    @property
    def rotate(self) -> npt.NDArray:
        """array: Rotation matrix in world coordinates."""
        return self._transforms['data2physical'].rotate

    @rotate.setter
    def rotate(self, rotate: npt.NDArray) -> None:
        self._transforms['data2physical'].rotate = rotate
        self._parameters_with_default_values.discard('rotate')
        self.events.rotate.emit(rotate)

    @property
    def scale(self) -> npt.NDArray:
        """array: Anisotropy factors to scale data into world coordinates."""
        return self._transforms['data2physical'].scale

    @scale.setter
    def scale(self, scale: Optional[npt.NDArray]) -> None:
        if scale is None:
            scale = np.array([1] * self.ndim)
        self._transforms['data2physical'].scale = np.array(scale)
        self._parameters_with_default_values.discard('scale')
        self.events.scale.emit(self.scale)

    @property
    def shear(self) -> npt.NDArray:
        """array: Shear matrix in world coordinates."""
        return self._transforms['data2physical'].shear

    @shear.setter
    def shear(self, shear: npt.NDArray) -> None:
        self._transforms['data2physical'].shear = shear
        self._parameters_with_default_values.discard('shear')
        self.events.shear.emit(self.shear)

    @property
    def translate(self) -> npt.NDArray:
        """array: Factors to shift the layer by in units of world coordinates."""
        return self._transforms['data2physical'].translate

    @translate.setter
    def translate(self, translate: npt.ArrayLike) -> None:
        self._transforms['data2physical'].translate = np.array(translate)
        self._parameters_with_default_values.discard('translate')
        self.events.translate.emit(self.translate)

    @property
    def ndim(self) -> int:
        """int: Number of dimensions."""
        return self._ndim

    @property
    def parameters_with_default_values(self) -> set[str]:
        """set[str]: Parameters that have default values
        (passed `None` to constructor and not set later).
        """
        return set(self._parameters_with_default_values)

    def set_axis_and_units(
        self, axes_labels: Sequence[str], units: UnitsLike
    ) -> None:
        self._units, self._axes_labels = _coerce_units_and_axes(
            units, axes_labels
        )
        self._parameters_with_default_values.discard('axes_labels')
        self._parameters_with_default_values.discard('units')
        self.events.axes_labels.emit(self.axes_labels)
        self.events.units.emit(self.units)

    @property
    def axes_labels(self) -> List[str]:
        """Sequence[str]: Labels for each axis."""
        if self._axes_labels is None:
            if self.ndim < 5:
                return ['t', 'z', 'y', 'x'][-self.ndim :]
            return [f'axis {i}' for i in range(self.ndim)][::-1]
        return self._axes_labels

    @axes_labels.setter
    def axes_labels(self, axes_labels: Sequence[str]) -> None:
        axes_labels = list(axes_labels)
        if len(axes_labels) != len(set(axes_labels)):
            raise ValueError('Axes labels must be unique.')
        if len(axes_labels) != self.ndim:
            raise ValueError(
                f'Length of axes_labels should be equal to ndim ({self.ndim})'
            )
        if isinstance(self._units, dict) and not set(axes_labels).issubset(
            set(self._units)
        ):
            diff = ', '.join(set(axes_labels) - set(self._units))
            raise ValueError(
                'Units are set per axis and some of new '
                'axes_labels do not have a corresponding unit. '
                f'Missing units for: {diff}. '
                'Please use set_axis_and_units method.'
            )
        self._axes_labels = list(axes_labels)
        self._parameters_with_default_values.discard('axes_labels')
        self.events.axes_labels.emit(self.axes_labels)

    @property
    def units(self) -> Dict[str, pint.Unit]:
        """Dict[str, unyt.Unit]: Units for each axis."""
        if isinstance(self._units, dict):
            return self._units
        return {label: self._units for label in self.axes_labels}

    @units.setter
    def units(self, units: UnitsLike) -> None:
        self._units, self._axes_labels = _coerce_units_and_axes(
            units, self._axes_labels
        )
        self._parameters_with_default_values.discard('units')
        self.events.units.emit(self.units)


def _coerce_units_and_axes(
    units: UnitsLike, axes_labels: Optional[Sequence[str]]
) -> Tuple[UnitsInfo, Optional[List[str]]]:
    units_ = _get_units_from_name(units)
    if axes_labels is None:
        return units_, None

    axes_labels = list(axes_labels)
    if len(axes_labels) != len(set(axes_labels)):
        raise ValueError('Axes labels must be unique.')

    if isinstance(units_, dict):
        if set(axes_labels).issubset(set(units_)):
            units_ = {name: units_[name] for name in axes_labels}
        else:
            diff = ', '.join(set(axes_labels) - set(units_))
            raise ValueError(
                'If both axes_labels and units are provided, '
                'all axes_labels must have a corresponding unit. '
                f'Missing units for: {diff}'
            )
    return units_, axes_labels


@overload
def _get_units_from_name(units: None) -> None: ...


@overload
def _get_units_from_name(units: Union[str, pint.Unit]) -> pint.Unit: ...


@overload
def _get_units_from_name(
    units: dict[str, Union[str, pint.Unit]]
) -> dict[str, pint.Unit]: ...


def _get_units_from_name(units: UnitsLike) -> UnitsInfo:
    """
    Convert a string or dict of strings to unyt units.
    """
    try:
        if isinstance(units, str):
            return pint.get_application_registry()[units].units
        if isinstance(units, dict):
            return {
                name: (
                    value
                    if isinstance(value, pint.Unit)
                    else pint.get_application_registry()[value].units
                )
                for name, value in units.items()
            }
    except AttributeError as e:
        raise ValueError(f'Could not find unit {units}') from e
    return units
