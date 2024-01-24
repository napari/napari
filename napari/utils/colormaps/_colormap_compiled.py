import logging

import numpy as np

try:
    from napari.utils.colormaps._colormap_numba import (
        labels_raw_to_texture_direct_numba,
        zero_preserving_modulo_numba,
    )

    NUMBA = True
except ImportError:
    NUMBA = False
    zero_preserving_modulo_numba = None
    labels_raw_to_texture_direct_numba = None

try:
    from napari.utils.colormaps._colormap_partseg import (
        labels_raw_to_texture_direct_partseg,
        zero_preserving_modulo_partseg,
    )

    PARTSEG = True
except ImportError:
    PARTSEG = False
    zero_preserving_modulo_partseg = None
    labels_raw_to_texture_direct_partseg = None

if not (NUMBA or PARTSEG):
    raise ImportError("No compiled backend available")

if PARTSEG:

    def zero_preserving_modulo(
        values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
    ) -> np.ndarray:
        try:
            return zero_preserving_modulo_partseg(values, n, dtype, to_zero)
        except Exception:
            if NUMBA:
                logging.exception("Error in PartSeg backend")
                return zero_preserving_modulo_numba(values, n, dtype, to_zero)
            raise

    def labels_raw_to_texture_direct(
        data: np.ndarray, direct_colormap
    ) -> np.ndarray:
        try:
            return labels_raw_to_texture_direct_partseg(data, direct_colormap)
        except Exception:
            if NUMBA:
                logging.exception("Error in PartSeg backend")
                return labels_raw_to_texture_direct_numba(
                    data, direct_colormap
                )
            raise

else:
    zero_preserving_modulo = zero_preserving_modulo_numba
    labels_raw_to_texture_direct = labels_raw_to_texture_direct_numba
