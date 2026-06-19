"""Debug script for progressive loading offset investigation.

Run:  .venv/bin/python examples/progressive_loading_zebrahub_debug_.py
"""

import logging

import numpy as np
import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, MemoryStore

logging.basicConfig(level=logging.WARNING)

import napari  # noqa: E402
from napari.experimental._progressive_loading import (  # noqa: E402
    add_progressive_loading_image,
)

URL = 'https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS002.ome.zarr/'
NUM_LEVELS = 4


def open_zebrahub():
    store = CacheStore(
        FsspecStore.from_url(URL),
        cache_store=MemoryStore(),
        max_size=int(4e9),
    )
    group = zarr.open_group(store, mode='r')
    arrays = [group[str(level)] for level in range(NUM_LEVELS)]
    ms = dict(group.attrs)['multiscales'][0]
    scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
    return arrays, scale


arrays, scale = open_zebrahub()
print(f"OME scale: {scale}")
for i, a in enumerate(arrays):
    print(f"  level {i}: shape={a.shape}  chunks={a.chunks}")

viewer = napari.Viewer()
layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    contrast_limits=(0, 1000),
    colormap='gray',
    scale=scale,
)

layer.bounding_box.visible = True

loader = layer.metadata['progressive_loader']

print(f"\nlayer.scale = {layer.scale}")
print(f"_max_tile_extent_3d = {layer._max_tile_extent_3d}")
print(f"level_shapes = {layer.level_shapes}")
print("downsample_factors:")
for i, ds in enumerate(layer.downsample_factors):
    print(f"  level {i}: {ds}")


def debug_state():
    cp = layer.corner_pixels
    level = layer.data_level
    ndisplay = viewer.dims.ndisplay
    if ndisplay != 3:
        return

    t2d = layer._transforms[0]
    t2d_translate = getattr(t2d, 'translate', None)

    ds = np.array(layer.downsample_factors[level])
    displayed = list(layer._slice_input.displayed)

    # tile center in level-0 data coords
    tile_center_lv = (cp[0, displayed] + cp[1, displayed]) / 2.0
    tile_center_lv0 = tile_center_lv * ds[displayed]

    # camera center in data coords
    cam = viewer.camera
    try:
        world_point = np.array(viewer.dims.point, dtype=float)
        world_point[displayed] = np.array(cam.center)[-len(displayed):]
        data_center = np.asarray(layer.world_to_data(world_point), dtype=float)
        cam_data = data_center[displayed]
    except Exception:  # noqa: BLE001
        cam_data = None

    offset = tile_center_lv0 - cam_data if cam_data is not None else None

    print(f"\n--- 3D state (level={level}) ---")
    print(f"  corners[0,disp]={cp[0, displayed]}  corners[1,disp]={cp[1, displayed]}")
    print(f"  tile_center_at_level={tile_center_lv}")
    print(f"  tile_center_lv0={tile_center_lv0}")
    print(f"  camera_data_center={cam_data}")
    print(f"  OFFSET (tile-camera in lv0)={offset}")
    print(f"  tile2data translate={t2d_translate}")
    print(f"  camera.center={cam.center}  zoom={cam.zoom:.3f}")


from qtpy.QtCore import QTimer  # noqa: E402

timer = QTimer()
timer.setInterval(3000)
timer.timeout.connect(debug_state)
timer.start()

print("\n>>> Bounding box ON. Debug timer 3s. _corners_for_locked_level logs via WARNING.")
print(">>> Switch to 3D to see offset diagnostics.\n")

if __name__ == '__main__':
    napari.run()
