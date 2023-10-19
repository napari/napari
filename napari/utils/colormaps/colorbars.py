from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from vispy.color import Colormap


def make_colorbar(
    cmap: 'Colormap', size: Tuple[int, int] = (18, 28), horizontal: bool = True
) -> npt.NDArray[np.uint8]:
    """Make a colorbar from a colormap.

    Parameters
    ----------
    cmap : vispy.color.Colormap
        Colormap to create colorbar with.
    size : 2-tuple
        Shape of colorbar.
    horizontal : bool
        If True colobar is oriented horizontal, otherwise it is oriented
        vertical.

    Returns
    -------
    cbar : array
        Array of colorbar in uint8.
    """

    if horizontal:
        basic_values = np.linspace(0, 1, size[1])
        bar = np.tile(np.expand_dims(basic_values, 1), size[0]).transpose(
            (1, 0)
        )
    else:
        basic_values = np.linspace(0, 1, size[0])
        bar = np.tile(np.expand_dims(basic_values, 1), size[1])

    color_array = cmap.map(bar.ravel())
    cbar = color_array.reshape((*bar.shape, 4))

    return np.round(255 * cbar).astype(np.uint8).copy(order='C')
