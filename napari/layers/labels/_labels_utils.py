from functools import lru_cache

import numpy as np


def interpolate_coordinates(old_coord, new_coord, brush_size):
    """Interpolates coordinates depending on brush size.

    Useful for ensuring painting is continuous in labels layer.

    Parameters
    ----------
    old_coord : np.ndarray, 1x2
        Last position of cursor.
    new_coord : np.ndarray, 1x2
        Current position of cursor.
    brush_size : float
        Size of brush, which determines spacing of interpolation.

    Returns
    -------
    coords : np.array, Nx2
        List of coordinates to ensure painting is continuous
    """
    num_step = round(
        max(abs(np.array(new_coord) - np.array(old_coord))) / brush_size * 4
    )
    coords = [
        np.linspace(old_coord[i], new_coord[i], num=int(num_step + 1))
        for i in range(len(new_coord))
    ]
    coords = np.stack(coords).T
    if len(coords) > 1:
        coords = coords[1:]

    return coords


@lru_cache(maxsize=64)
def sphere_indices(radius, sphere_dims):
    """Generate centered indices within circle or n-dim sphere.

    Parameters
    -------
    radius : float
        Radius of circle/sphere
    sphere_dims : int
        Number of circle/sphere dimensions

    Returns
    -------
    mask_indices : array
        Centered indices within circle/sphere
    """
    # Create multi-dimensional grid to check for
    # circle/membership around center
    vol_radius = radius + 0.5

    indices_slice = [slice(-vol_radius, vol_radius + 1)] * sphere_dims
    indices = np.mgrid[indices_slice].T.reshape(-1, sphere_dims)
    distances_sq = np.sum(indices ** 2, axis=1)
    # Use distances within desired radius to mask indices in grid
    mask_indices = indices[distances_sq <= radius ** 2].astype(int)

    return mask_indices
