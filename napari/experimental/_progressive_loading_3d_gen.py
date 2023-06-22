import logging
import sys
import time

import dask.array as da
import numpy as np
import xarray
import zarr
from fibsem_tools import read_xarray
from numba import njit
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from zarr.storage import init_array, init_group
from zarr.util import json_dumps

def create_meta_store(levels, tilesize, compressor, dtype):
    store = dict()
    init_group(store)

    datasets = [{"path": str(i)} for i in range(levels)]
    root_attrs = {"multiscales": [{"datasets": datasets, "version": "0.1"}]}
    store[".zattrs"] = json_dumps(root_attrs)

    base_width = tilesize * 2**levels
    for level in range(levels):
        width = int(base_width / 2**level)
        init_array(
            store,
            path=str(level),
            shape=(width, width, width),
            chunks=(tilesize, tilesize, tilesize),
            dtype=dtype,
            compressor=compressor,
        )
    return store

# @njit(nogil=True)
# def mandelbulb(out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter):
#     step_x = (to_x - from_x) / grid_size
#     step_y = (to_y - from_y) / grid_size
#     step_z = (to_z - from_z) / grid_size
#     creal = from_x
#     cimag = from_y
#     cimag2 = from_z
#     for i in range(grid_size):
#         cimag = from_y
#         for j in range(grid_size):
#             cimag2 = from_z
#             for k in range(grid_size):
#                 nreal = real = imag = imag2 = n = 0
#                 # Use Cardioid / bulb checking for early termination
#                 q = (i - 0.25) ** 2 + j**2 + k**2
#                 if q * (q + (i - 0.25)) > 0.25 * (j**2 + k**2):
#                     for _ in range(maxiter):
#                         nreal = real * real - imag * imag - imag2 * imag2 + creal
#                         imag = 2 * real * imag + cimag
#                         imag2 = 2 * real * imag2 + cimag2
#                         real = nreal
#                         if real * real + imag * imag + imag2 * imag2 > 4.0:
#                             break
#                         n += 1
#                 out[k * grid_size * grid_size + j * grid_size + i] = n
#                 cimag2 += step_z
#             cimag += step_y
#         creal += step_x

#     return out

# @njit(nogil=True)
# def mandelbulb(out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter):
#     step_x = (to_x - from_x) / grid_size
#     step_y = (to_y - from_y) / grid_size
#     step_z = (to_z - from_z) / grid_size
#     creal = from_x
#     cimag = from_y
#     cimag2 = from_z
#     for i in range(grid_size):
#         cimag = from_y
#         for j in range(grid_size):
#             cimag2 = from_z
#             for k in range(grid_size):
#                 nreal = real = imag = imag2 = n = 0
#                 for _ in range(maxiter):
#                     nreal = real * real - imag * imag - imag2 * imag2 + creal
#                     imag = 2 * real * imag + cimag
#                     imag2 = 2 * real * imag2 + cimag2
#                     real = nreal
#                     if real * real + imag * imag + imag2 * imag2 > 4.0:
#                         break
#                     n += 1
#                 out[k * grid_size * grid_size + j * grid_size + i] = n
#                 cimag2 += step_z
#             cimag += step_y
#         creal += step_x

#     return out

# This is the boring mandelbrot with extra axis version
# @njit(nogil=True)
# def mandelbulb(out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter):
#     step_x = (to_x - from_x) / grid_size
#     step_y = (to_y - from_y) / grid_size
#     step_z = (to_z - from_z) / grid_size
#     creal = from_x
#     cimag = from_y
#     cimag2 = from_z
#     for i in range(grid_size):
#         cimag = from_y
#         for j in range(grid_size):
#             cimag2 = from_z
#             for k in range(grid_size):
#                 nreal = real = imag = imag2 = n = 0
#                 for _ in range(maxiter):
#                     nreal = real * real - imag * imag - imag2 * imag2 + creal
#                     imag = 2 * real * imag + cimag
#                     imag2 = 2 * real * imag2 + cimag2
#                     real = nreal
#                     if real * real + imag * imag + imag2 * imag2 > 4.0:
#                         break
#                     out[i * grid_size * grid_size + j * grid_size + k] = n  # Modify indexing
#                     n += 1
#                 cimag2 += step_z
#             cimag += step_y
#         creal += step_x

#     return out


# Looks buggy
# @njit(nogil=True)
# def mandelbulb(out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter):
#     step_x = (to_x - from_x) / grid_size
#     step_y = (to_y - from_y) / grid_size
#     step_z = (to_z - from_z) / grid_size
#     creal = from_x
#     cimag = from_y
#     cimag2 = from_z
#     for i in range(grid_size):
#         cimag = from_y
#         for j in range(grid_size):
#             cimag2 = from_z
#             for k in range(grid_size):
#                 nreal = real = imag = imag2 = n = 0
#                 zx = zy = zz = 0  # Initialize z components
#                 for _ in range(maxiter):
#                     r2 = zx * zx + zy * zy + zz * zz
#                     if r2 > 4.0:  # Check magnitude squared against threshold
#                         break
#                     z8 = zx * zx * zx * zx * zx * zx * zx * zx  # Compute z⁸
#                     nreal = z8 - 28.0 * zx * zx * zx * zy * zy * zz - 14.0 * zx * zx * zz * zz + creal
#                     imag = 8.0 * zx * zx * zx * zx * zy * zz - 14.0 * z8 + 8.0 * zx * zy * zy * zy * zz + cimag
#                     imag2 = -8.0 * zx * zx * zx * zx * zx * zy + 8.0 * zx * zx * zy * zy * zy + cimag2
#                     zx = nreal
#                     zy = imag
#                     zz = imag2
#                     n += 1
#                 out[i * grid_size * grid_size + j * grid_size + k] = n  # Modify indexing
#                 cimag2 += step_z
#             cimag += step_y
#         creal += step_x

#     return out


# @njit(nogil=True)
# def mandelbulb(out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter):
#     step_x = (to_x - from_x) / grid_size
#     step_y = (to_y - from_y) / grid_size
#     step_z = (to_z - from_z) / grid_size
#     creal = from_x
#     cimag = from_y
#     cimag2 = from_z
#     for i in range(grid_size):
#         cimag = from_y
#         for j in range(grid_size):
#             cimag2 = from_z
#             for k in range(grid_size):
#                 nreal = real = imag = imag2 = n = 0
#                 zx = zy = zz = 0  # Initialize z components
#                 for _ in range(maxiter):
#                     r2 = zx * zx + zy * zy + zz * zz
#                     if r2 > 4.0:  # Check magnitude squared against threshold
#                         break
#                     x = zx
#                     y = zy
#                     z = zz

#                     # Quintic formula
#                     x2 = x * x
#                     y2 = y * y
#                     z2 = z * z
#                     x4 = x2 * x2
#                     y4 = y2 * y2
#                     z4 = z2 * z2

#                     x8 = x4 * x4
#                     y8 = y4 * y4
#                     z8 = z4 * z4

#                     A = 10 * (y2 * z + 0.1 * x * z2)
#                     B = 5 * (y4 + 0.1 * y * y * z + 0.1 * y * z * z2 + x4 + 0.1 * x * x * x * z + 0.1 * x * z * z2)
#                     C = 5 * (y * y * y * z + y * z * z2 + 0.1 * x2 * z2)
#                     D = 0.1 * x2 * y * z * (y + z)

#                     nreal = x8 - 10 * x4 * (y2 + A) + 5 * x * (y4 + B) + D
#                     imag = y8 - 10 * y4 * (z2 + A) + 5 * y * (z4 + B) + D
#                     imag2 = z8 - 10 * z4 * (x2 + A) + 5 * z * (x4 + B) + D

#                     zx = nreal + creal
#                     zy = imag + cimag
#                     zz = imag2 + cimag2
#                     n += 1
#                 out[i * grid_size * grid_size + j * grid_size + k] = n  # Modify indexing
#                 cimag2 += step_z
#             cimag += step_y
#         creal += step_x

#     return out


# Based on http://www.fractal.org/Formula-Mandelbulb.pdf
@njit(nogil=True)
def hypercomplex_exponentiation(x, y, z, n):
    r = np.sqrt(x*x + y*y + z*z)
    r1 = np.sqrt(x*x + y*y)
    theta = np.arctan2(z, r1)
    phi = np.arctan2(y, x)
    new_r = r**n
    new_x = new_r * np.cos(n*phi) * np.cos(n*theta)
    new_y = new_r * np.sin(n*phi) * np.cos(n*theta)
    new_z = new_r * np.sin(n*theta)
    return new_x, new_y, new_z

@njit(nogil=True)
def mandelbulb(out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter, n):
    step_x = (to_x - from_x) / grid_size
    step_y = (to_y - from_y) / grid_size
    step_z = (to_z - from_z) / grid_size
    
    for i in range(grid_size):
        creal = from_x + i * step_x
        for j in range(grid_size):
            cimag = from_y + j * step_y
            for k in range(grid_size):
                cimag2 = from_z + k * step_z
                nreal = real = imag = imag2 = n_iter = 0
                for _ in range(maxiter):
                    nreal, nimag, nimag2 = hypercomplex_exponentiation(real, imag, imag2, n)
                    nreal += creal
                    nimag += cimag
                    nimag2 += cimag2
                    real = nreal
                    imag = nimag
                    imag2 = nimag2
                    if real*real + imag*imag + imag2*imag2 > 4.0:
                        break
                    out[i * grid_size * grid_size + j * grid_size + k] = n_iter
                    n_iter += 1

    return out

@njit(nogil=True)
def tile_bounds(level, x, y, z, max_level, min_coord=-2.5, max_coord=2.5):
    max_width = max_coord - min_coord
    tile_width = max_width / 2 ** (max_level - level)
    from_x = min_coord + x * tile_width
    to_x = min_coord + (x + 1) * tile_width

    from_y = min_coord + y * tile_width
    to_y = min_coord + (y + 1) * tile_width

    from_z = min_coord + z * tile_width
    to_z = min_coord + (z + 1) * tile_width

    return from_x, from_y, from_z, to_x, to_y, to_z


class MandlebulbStore(zarr.storage.Store):
    def __init__(self, levels, tilesize, maxiter=255, compressor=None):
        self.levels = levels
        self.tilesize = tilesize
        self.compressor = compressor
        self.dtype = np.dtype(np.uint8 if maxiter < 256 else np.uint16)
        self.maxiter = maxiter
        self.order = 4
        self._store = create_meta_store(
            levels, tilesize, compressor, self.dtype
        )

    def __getitem__(self, key):
        if key in self._store:
            return self._store[key]

        try:
            # Try parsing pyramidal coords
            level, chunk_key = key.split("/")
            level = int(level)
            z, y, x = map(int, chunk_key.split("."))
        except:
            raise KeyError

        return self.get_chunk(level, z, y, x).tobytes()

    def get_chunk(self, level, z, y, x):
        from_x, from_y, from_z, to_x, to_y, to_z = tile_bounds(level, x, y, z, self.levels)
        out = np.zeros(self.tilesize * self.tilesize * self.tilesize, dtype=self.dtype)
        tile = mandelbulb(
            out,
            from_x,
            from_y,
            from_z,
            to_x,
            to_y,
            to_z,
            self.tilesize,
            self.maxiter,
            self.order,
        )
        tile = tile.reshape(self.tilesize, self.tilesize, self.tilesize).transpose()

        if self.compressor:
            return self.compressor.encode(tile)

        return tile

    def keys(self):
        return self._store.keys()

    def __iter__(self):
        return iter(self._store)

    def __delitem__(self, key):
        if key in self._store:
            del self._store[key]

    def __len__(self):
        return len(self._store)  # TODO not correct

    def __setitem__(self, key, val):
        self._store[key] = val
        


# https://dask.discourse.group/t/using-da-delayed-for-zarr-processing-memory-overhead-how-to-do-it-better/1007/10
def mandelbulb_dataset(max_levels=14):
    """Generate a multiscale image of the mandelbulb set for a given number
    of levels/scales. Scale 0 will be the highest resolution.

    This is intended to be used with progressive loading. As such, it returns
    a dictionary will all the metadata required to load as multiple scaled
    image layers via add_progressive_loading_image

    >>> large_image = mandelbrot_dataset(max_levels=14)
    >>> multiscale_img = large_image["arrays"]
    >>> viewer._layer_slicer._force_sync = False
    >>> add_progressive_loading_image(multiscale_img, viewer=viewer)

    Parameters
    ----------
    max_levels: int
        Maximum number of levels (scales) to generate

    Returns
    -------
    Dictionary of multiscale data with keys ['container', 'dataset',
        'scale levels', 'scale_factors', 'chunk_size', 'arrays']
    """

    large_image = {
        "container": "mandelbulb.zarr/",
        "dataset": "",
        "scale_levels": max_levels,
        "scale_factors": [
            (2**level, 2**level, 2**level) for level in range(max_levels)
        ],
        "chunk_size": (32, 32, 32),
    }

    # Initialize the store
    store = zarr.storage.KVStore(
        MandlebulbStore(
            levels=max_levels,
            tilesize=32,
            compressor=None,
            maxiter=255
            #        levels=max_levels, tilesize=512, compressor=Blosc(), maxiter=255
        )
    )
    # Wrap in a cache so that tiles don't need to be computed as often
    # store = zarr.LRUStoreCache(store, max_size=8e9)

    # This store implements the 'multiscales' zarr specfiication which is recognized by vizarr
    z_grp = zarr.open(store, mode="r")

    multiscale_img = [z_grp[str(k)] for k in range(max_levels)]

    arrays = []
    for scale, a in enumerate(multiscale_img):

        chunks = da.core.normalize_chunks(
            large_image["chunk_size"],
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        # arrays += [da.from_zarr(a, chunks=chunks)]

        setattr(
            a,
            "get_zarr_chunk",
            lambda scale, z, y, x: store.get_chunk(scale, z, y, x),
        )
        # setattr(a, "get_zarr_chunk", lambda chunk_slice: a[tuple(chunk_slice)].transpose())
        arrays += [a]

    large_image["arrays"] = arrays

    # TODO wrap in dask delayed

    return large_image


def point_in_bounding_box(point, bounding_box):
    if np.all(point > bounding_box[0]) and np.all(point < bounding_box[1]):
        return True
    return False


import napari

viewer = napari.Viewer(ndisplay=3)

@viewer.mouse_drag_callbacks.append
def shift_plane_along_normal(viewer, event):
    """Shift a plane along its normal vector on mouse drag.

    This callback will shift a plane along its normal vector when the plane is
    clicked and dragged. The general strategy is to
    1) find both the plane normal vector and the mouse drag vector in canvas
    coordinates
    2) calculate how far to move the plane in canvas coordinates, this is done
    by projecting the mouse drag vector onto the (normalised) plane normal
    vector
    3) transform this drag distance (canvas coordinates) into data coordinates
    4) update the plane position

    It will also add a point to the points layer for a 'click-not-drag' event.
    """
    # get layers from viewer
    volume_layer = viewer.layers['volume']

    # Calculate intersection of click with data bounding box
    near_point, far_point = volume_layer.get_ray_intersections(
        event.position,
        event.view_direction,
        event.dims_displayed,
    )

    # Calculate intersection of click with plane through data
    intersection = volume_layer.experimental_clipping_planes[0].intersect_with_line(
        line_position=near_point, line_direction=event.view_direction
    )

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If so, exit early.
    if not point_in_bounding_box(intersection, volume_layer.extent.data):
        print('not in bounding box')
        return

    print(f"in bounding box {intersection}")
    
    volume_layer = viewer.layers["volume"]
    
    # Get plane parameters in vispy coordinates (zyx -> xyz)
    plane_normal_data_vispy = np.array(volume_layer.experimental_clipping_planes[0].normal)[[2, 1, 0]]
    plane_position_data_vispy = np.array(volume_layer.experimental_clipping_planes[0].position)[[2, 1, 0]]

    # Get transform which maps from data (vispy) to canvas
    visual2canvas = viewer.window.qt_viewer.layer_to_visual[volume_layer].node.get_transform(
        map_from="visual", map_to="canvas"
    )

    # Find start and end positions of plane normal in canvas coordinates
    plane_normal_start_canvas = visual2canvas.map(plane_position_data_vispy)
    plane_normal_end_canvas = visual2canvas.map(plane_position_data_vispy + plane_normal_data_vispy)

    # Calculate plane normal vector in canvas coordinates
    plane_normal_canv = (plane_normal_end_canvas - plane_normal_start_canvas)[[0, 1]]
    plane_normal_canv_normalised = (
            plane_normal_canv / np.linalg.norm(plane_normal_canv)
    )

    # Disable interactivity during plane drag
    volume_layer.interactive = False

    # Store original plane position and start position in canvas coordinates
    original_plane_position = volume_layer.experimental_clipping_planes[0].position
    start_position_canv = event.pos

    yield
    while event.type == "mouse_move":
        # Get end position in canvas coordinates
        end_position_canv = event.pos

        # Calculate drag vector in canvas coordinates
        drag_vector_canv = end_position_canv - start_position_canv

        # Project the drag vector onto the plane normal vector
        # (in canvas coorinates)
        drag_projection_on_plane_normal = np.dot(
            drag_vector_canv, plane_normal_canv_normalised
        )

        # Update position of plane according to drag vector
        # only update if plane position is within data bounding box
        drag_distance_data = drag_projection_on_plane_normal / np.linalg.norm(plane_normal_canv)
        updated_position = original_plane_position + drag_distance_data * np.array(
            volume_layer.experimental_clipping_planes[0].normal)

        if point_in_bounding_box(updated_position, volume_layer.extent.data):
            volume_layer.experimental_clipping_planes[0].position = updated_position

        yield

    # Re-enable
    volume_layer.interactive = True

if __name__ == "__main__":

    plane_parameters = {
        'position': (128, 128, 128),
        'normal': (1, 1, 1),
        'enabled': True
    }
    
    large_image = mandelbulb_dataset(max_levels=3)

    multiscale_img = large_image["arrays"]

    print(multiscale_img[0])
    
    layer = viewer.add_image(multiscale_img[0], name="volume", experimental_clipping_planes=[plane_parameters])
    layer.bounding_box.visible = True
   
    viewer.axes.visible = True
    viewer.camera.angles = (45, 45, 45)
    viewer.camera.zoom = 5
    viewer.text_overlay.update(dict(
        text='Click and drag the clipping plane surface to move it along its normal.',
        visible=True,
    ))
