from os import path

import numpy as np

from  napari.utils.notifications import show_warning
from napari.utils._platformdirs import user_cache_dir

try:
    from numba import njit
    is_numba = True
except ImportError:
    is_numba = False
    def njit(func):
        return func
    
DATA_NAME = 'cylinder_difusion.tiff'


@njit
def process_diffusion(dt, alpha, initial_state, t_max, n_snapshots):
    alpha_dt = alpha * dt

    u = np.copy(initial_state)

    nz, ny, nx = u.shape

    it_max = int(t_max / dt)
    
    n_frames = (it_max - 1) // n_snapshots + 1

    evolution = np.zeros((n_frames, nz, ny, nx), dtype=u.dtype)

    it = 0
    frame = 0

    while it < it_max:

        u_new = np.copy(u)

        for i in range(1, nz - 1):
            for j in range(1, ny - 1):
                for k in range(1, nx - 1):
                    if not np.isnan(u[i, j, k]):
                        u_dz_p = u[i, j, k] if np.isnan(u[i + 1, j, k]) else u[i + 1, j, k]
                        u_dz_n = u[i, j, k] if np.isnan(u[i - 1, j, k]) else u[i - 1, j, k]

                        u_dy_p = u[i, j, k] if np.isnan(u[i, j + 1, k]) else u[i, j + 1, k]
                        u_dy_n = u[i, j, k] if np.isnan(u[i, j - 1, k]) else u[i, j - 1, k]

                        u_dx_p = u[i, j, k] if np.isnan(u[i, j, k + 1]) else u[i, j, k + 1]
                        u_dx_n = u[i, j, k] if np.isnan(u[i, j, k - 1]) else u[i, j, k - 1]

                        laplacian = u_dz_p + u_dz_n + u_dy_p + u_dy_n + u_dx_p + u_dx_n - 6 * u[i, j, k]

                        u_new[i, j, k] = u[i, j, k] + alpha_dt * laplacian

        if it % n_snapshots == 0:
            evolution[frame] = u_new
            frame += 1

        u = u_new

        it += 1

    return evolution


def cylinder_diffusion():
    # example_data = np.random.randint(0, 255, size=(50, 64, 32, 32))

    nz, nx, ny = 64, 32, 32
    z, y, x = np.ogrid[:nz, :ny, :nx]

    x_center = nx // 2
    y_center = ny // 2

    radius = 12

    cylinder_mask = (y - y_center)**2 + (x - x_center)**2 <= radius**2
    cylinder_volume = np.zeros((nz, ny, nx))
    x_heat = nx // 3
    y_heat = ny // 2
    z_heat = nz // 5
    r_heat = 6

    cylinder_heat_mask = (z - z_heat)**2 + (y - y_heat)**2 + (x - x_heat)**2 <= r_heat**2

    for i in range(1, nz - 1):
        cylinder_volume[i, cylinder_mask[0]] = 10.0

    cylinder_volume[np.logical_and(cylinder_mask, cylinder_heat_mask)] = 1000.0

    cylinder_volume = np.where(cylinder_volume == 0, np.nan, cylinder_volume)

    process_diffusion(1, 1, cylinder_volume, 1, 1)

    evolution = process_diffusion(0.1, 0.8, cylinder_volume, 500.0, 50)

    evolution = np.nan_to_num(evolution)
    return evolution


def cyclic_difusion_():
    # cache
    evolution = cylinder_diffusion()
    # check if there si numba, otherwise show warnig 
    return [
        (
            evolution,
            {
                'name': 'heat_diffusion',
                'metadata': {'axes': ['t', 'z', 'y', 'x']},
                'colormap': 'viridis'
            },
            'image'
        )
    ]