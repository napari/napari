from typing import TYPE_CHECKING

from lazy_loader import attach as _attach

from napari._check_numpy_version import NUMPY_VERSION_IS_THREADSAFE
from napari.utils._dask_utils import resize_dask_cache
from napari.utils.colormaps.colormap import (
    Colormap,
    CyclicLabelColormap,
    DirectLabelColormap,
)
from napari.utils.info import citation_text, sys_info
from napari.utils.notebook_display import (
    NotebookScreenshot,
    nbscreenshot,
)
from napari.utils.progress import cancelable_progress, progrange, progress

if TYPE_CHECKING:
    from napari.utils._remote_zarr import open_cached_zarr

__all__ = (
    'NUMPY_VERSION_IS_THREADSAFE',
    'Colormap',
    'CyclicLabelColormap',
    'DirectLabelColormap',
    'NotebookScreenshot',
    'cancelable_progress',
    'citation_text',
    'nbscreenshot',
    'open_cached_zarr',
    'progrange',
    'progress',
    'resize_dask_cache',
    'sys_info',
)

# zarr is an optional dependency, so its module is imported on first access
__getattr__ = _attach(
    __name__, submod_attrs={'_remote_zarr': ['open_cached_zarr']}
)[0]
del _attach
