import numpy as np


def make_colorbar(cmap, size=(12, 28), horizontal=True):
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
    ----------
    cbar : array
        Array of colorbar.
    """

    if horizontal:
        input = np.linspace(0, 1, size[1])
        bar = np.tile(np.expand_dims(input, 1), size[0]).transpose((1, 0))
    else:
        input = np.linspace(0, 1, size[0])
        bar = np.tile(np.expand_dims(input, 1), size[1])

    # cmap.__getitem__ returns a vispy.color.ColorArray
    color_array = cmap[bar.ravel()]
    # the ColorArray.RGBA method returns a normalized uint8 array
    cbar = color_array.RGBA.reshape(bar.shape + (4,))

    return cbar
