from os import makedirs, path

import numpy as np
import tifffile

from napari.utils._platformdirs import user_cache_dir
from napari.utils.notifications import show_warning

try:
    from numba import njit

    is_numba = True
except ImportError:
    is_numba = False

    def njit(func):
        return func


DATA_NAME = 'cylinder_diffusion.tif'
CACHE_DIR = user_cache_dir()


@njit
def _neighbor_value(u, i, j, k, di, dj, dk):
    """Get neighbor value, using center value if neighbor is NaN."""
    neighbor = u[i + di, j + dj, k + dk]
    if np.isnan(neighbor):
        return u[i, j, k]
    return neighbor


@njit
def process_diffusion(dt, alpha, initial_state, t_max, n_snapshots):
    """Solver for 3D heat diffusion using FTCS method."""
    alpha_dt = alpha * dt
    u = np.copy(initial_state)

    it_max = int(t_max / dt)
    n_frames = (it_max - 1) // n_snapshots + 1
    evolution = np.zeros((n_frames, *u.shape), dtype=np.float64)

    frame = 0
    for it in range(it_max):
        u_new = np.copy(u)

        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                for k in range(1, u.shape[2] - 1):
                    if not np.isnan(u[i, j, k]):
                        laplacian = (
                            _neighbor_value(u, i, j, k, 1, 0, 0)
                            + _neighbor_value(u, i, j, k, -1, 0, 0)
                            + _neighbor_value(u, i, j, k, 0, 1, 0)
                            + _neighbor_value(u, i, j, k, 0, -1, 0)
                            + _neighbor_value(u, i, j, k, 0, 0, 1)
                            + _neighbor_value(u, i, j, k, 0, 0, -1)
                            - 6 * u[i, j, k]
                        )
                        u_new[i, j, k] = u[i, j, k] + alpha_dt * laplacian

        if it % n_snapshots == 0:
            evolution[frame] = u_new
            frame += 1

        u = u_new

    return evolution


def initial_cylinder_state():
    """Creates the initial state of temperature distribution over cylinder geometry."""
    nz, nx, ny = 64, 32, 32
    z, y, x = np.ogrid[:nz, :ny, :nx]

    x_center = nx // 2
    y_center = ny // 2

    radius = 12

    cylinder_mask = (y - y_center) ** 2 + (x - x_center) ** 2 <= radius**2
    cylinder_volume = np.zeros((nz, ny, nx))
    x_heat = nx // 3
    y_heat = ny // 2
    z_heat = nz // 5
    r_heat = 6

    cylinder_heat_mask = (z - z_heat) ** 2 + (y - y_heat) ** 2 + (
        x - x_heat
    ) ** 2 <= r_heat**2

    for i in range(1, nz - 1):
        cylinder_volume[i, cylinder_mask[0]] = 10.0

    cylinder_volume[np.logical_and(cylinder_mask, cylinder_heat_mask)] = 1000.0

    cylinder_volume = np.where(cylinder_volume == 0, np.nan, cylinder_volume)

    return cylinder_volume


def simulate_diffusion():
    initial_cylinder = initial_cylinder_state()

    # Numba warm-up call
    process_diffusion(1.0, 1.0, initial_cylinder, 1.0, 1)

    evolution = process_diffusion(0.1, 0.8, initial_cylinder, 500.0, 50)

    evolution = np.nan_to_num(evolution).astype(np.float32)

    return evolution


def cylinder_diffusion():
    """
    Loads the heat diffusion sample from cache if exists.
    Otherwise checks if numba is installed and runs the simulation.
    """
    makedirs(CACHE_DIR, exist_ok=True)

    data_path = path.join(CACHE_DIR, DATA_NAME)

    if path.exists(data_path):
        evolution = tifffile.imread(data_path)
    else:
        if is_numba:
            evolution = simulate_diffusion()

            tifffile.imwrite(data_path, evolution, compression='zlib')

        else:
            show_warning(
                'Numba is not installed but required for this sample data.'
            )
            return []

    return [
        (
            evolution,
            {
                'name': 'heat_diffusion',
                'metadata': {'axes': ['t', 'z', 'y', 'x']},
                'colormap': 'plasma',
            },
            'image',
        )
    ]
