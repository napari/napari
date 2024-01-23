from typing import TYPE_CHECKING

import numpy as np
from PartSegCore_compiled_backend.napari_mapping import (
    map_array_parallel,
    zero_preserving_modulo_parallel,
)

from napari.utils.colormaps._colormap_numpy import (
    MAPPING_OF_UNKNOWN_VALUE,
    minimum_dtype_for_labels,
)

if TYPE_CHECKING:
    from napari.utils import DirectLabelColormap

__all__ = ('zero_preserving_modulo', 'labels_raw_to_texture_direct')


def zero_preserving_modulo(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    out = np.empty_like(values, dtype=dtype)
    zero_preserving_modulo_parallel(values, n, to_zero, out)
    return out


def labels_raw_to_texture_direct(
    data: np.ndarray, direct_colormap: 'DirectLabelColormap'
) -> np.ndarray:
    dkt = direct_colormap._get_typed_dict_mapping(data.dtype)
    target_dtype = minimum_dtype_for_labels(
        direct_colormap._num_unique_colors + 2
    )
    out = np.empty_like(data, dtype=target_dtype)
    map_array_parallel(data, dkt, MAPPING_OF_UNKNOWN_VALUE, out)
    return out
