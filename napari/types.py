from functools import partial, wraps
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
from typing_extensions import TypedDict, get_args

if TYPE_CHECKING:
    import dask.array
    import zarr
    from magicgui.widgets import FunctionGui
    from qtpy.QtWidgets import QWidget


# This is a WOEFULLY inadequate stub for a duck-array type.
# Mostly, just a placeholder for the concept of needing an ArrayLike type.
# Ultimately, this should come from https://github.com/napari/image-types
# and should probably be replaced by a typing.Protocol
# note, numpy.typing.ArrayLike (in v1.20) is not quite what we want either,
# since it includes all valid arguments for np.array() ( int, float, str...)
ArrayLike = Union[np.ndarray, 'dask.array.Array', 'zarr.Array']


# layer data may be: (data,) (data, meta), or (data, meta, layer_type)
# using "Any" for the data type until ArrayLike is more mature.
FullLayerData = Tuple[Any, Dict, str]
LayerData = Union[Tuple[Any], Tuple[Any, Dict], FullLayerData]

PathLike = Union[str, Path]
PathOrPaths = Union[str, Sequence[str]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]
WriterFunction = Callable[[str, List[FullLayerData]], List[str]]

ExcInfo = Union[
    Tuple[Type[BaseException], BaseException, TracebackType],
    Tuple[None, None, None],
]

# Types for GUI HookSpecs
WidgetCallable = Callable[..., Union['FunctionGui', 'QWidget']]
AugmentedWidget = Union[WidgetCallable, Tuple[WidgetCallable, dict]]


# Sample Data for napari_provide_sample_data hookspec is either a string/path
# or a function that returns an iterable of LayerData tuples
SampleData = Union[PathLike, Callable[..., Iterable[LayerData]]]


# or... they can provide a dict as follows:
class SampleDict(TypedDict):
    display_name: str
    data: SampleData


# these types are mostly "intentionality" placeholders.  While it's still hard
# to use actual types to define what is acceptable data for a given layer,
# these types let us point to a concrete namespace to indicate "this data is
# intended to be (and is capable of) being turned into X layer type".
# while their names should not change (without deprecation), their typing
# implementations may... or may be rolled over to napari/image-types

ArrayBase = np.ndarray


ImageData = NewType("ImageData", ArrayBase)
LabelsData = NewType("LabelsData", ArrayBase)
PointsData = NewType("PointsData", ArrayBase)
ShapesData = NewType("ShapesData", List[ArrayBase])
SurfaceData = NewType("SurfaceData", Tuple[ArrayBase, ArrayBase, ArrayBase])
TracksData = NewType("TracksData", ArrayBase)
VectorsData = NewType("VectorsData", ArrayBase)
_LayerData = Union[
    ImageData,
    LabelsData,
    PointsData,
    ShapesData,
    SurfaceData,
    TracksData,
    VectorsData,
]

LayerDataTuple = NewType("LayerDataTuple", tuple)


def image_reader_to_layerdata_reader(
    func: Callable[[PathOrPaths], ArrayLike]
) -> ReaderFunction:
    """Convert a PathLike -> ArrayLike function to a PathLike -> LayerData.

    Parameters
    ----------
    func : Callable[[PathLike], ArrayLike]
        A function that accepts a string or list of strings, and returns an
        ArrayLike.

    Returns
    -------
    reader_function : Callable[[PathLike], List[LayerData]]
        A function that accepts a string or list of strings, and returns data
        as a list of LayerData: List[Tuple[ArrayLike]]

    """

    @wraps(func)
    def reader_function(*args, **kwargs) -> List[LayerData]:
        result = func(*args, **kwargs)
        return [(result,)]

    return reader_function


def _register_types_with_magicgui():
    """Register ``napari.types`` objects with magicgui."""
    import sys
    from concurrent.futures import Future

    from magicgui import register_type

    from napari.utils import _magicgui as _mgui

    for _type in (LayerDataTuple, List[LayerDataTuple]):
        register_type(
            _type,
            return_callback=_mgui.add_layer_data_tuples_to_viewer,
        )
        if sys.version_info >= (3, 9):
            future_type = Future[_type]  # type: ignore
            register_type(future_type, return_callback=_mgui.add_future_data)

    for data_type in get_args(_LayerData):
        register_type(
            data_type,
            choices=_mgui.get_layers_data,
            return_callback=_mgui.add_layer_data_to_viewer,
        )
        if sys.version_info >= (3, 9):
            register_type(
                Future[data_type],  # type: ignore
                choices=_mgui.get_layers_data,
                return_callback=partial(
                    _mgui.add_future_data, _from_tuple=False
                ),
            )
        register_type(
            Optional[data_type],  # type: ignore
            choices=_mgui.get_layers_data,
            return_callback=_mgui.add_layer_data_to_viewer,
        )
        if sys.version_info >= (3, 9):
            register_type(
                Future[Optional[data_type]],  # type: ignore
                choices=_mgui.get_layers_data,
                return_callback=partial(
                    _mgui.add_future_data, _from_tuple=False
                ),
            )


_register_types_with_magicgui()
