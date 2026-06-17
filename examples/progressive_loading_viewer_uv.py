# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "napari[pyqt5]",
#     "zarr>=3.1.6",
#     "fsspec>=2023.10.0",
#     "aiohttp",
#     "requests",
#     "s3fs",
# ]
#
# [tool.uv.sources]
# napari = { git = "https://github.com/kephale/napari", branch = "progressive-loading-rebase" }
# ///
"""
Progressive loading: generic OME-Zarr viewer
============================================

Open any local or remote OME-Zarr with progressive loading::

    uv run progressive_loading_viewer_uv.py /path/to/data.ome.zarr
    uv run progressive_loading_viewer_uv.py https://example.com/data.ome.zarr

Options::

    --levels N        Number of multiscale levels to load (default: all)
    --contrast LO HI  Contrast limits (default: auto-estimated)
    --colormap NAME   Colormap name (default: gray)
    --cache-mb N      In-memory cache size in MB for remote stores (default: 4000)
    --3d              Start in 3D display mode

.. tags:: experimental
"""

import argparse

import zarr

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)


def _is_remote(path: str) -> bool:
    return path.startswith(('http://', 'https://', 's3://', 'gs://'))


def _find_multiscales(group, depth=0):
    """Walk into nested groups to find OME multiscales metadata."""
    attrs = dict(group.attrs)
    ms_list = attrs.get('multiscales') or attrs.get('ome', {}).get('multiscales')
    if ms_list:
        return group, ms_list[0]
    if depth > 4:
        return None, None
    for key in group:
        try:
            child = group[key]
        except (KeyError, ValueError, OSError):
            continue
        if not hasattr(child, 'attrs'):
            continue
        result, ms = _find_multiscales(child, depth + 1)
        if ms is not None:
            return result, ms
    return None, None


def open_ome_zarr(path: str, *, num_levels: int | None = None, cache_mb: int = 4000):
    if _is_remote(path):
        from zarr.experimental.cache_store import CacheStore
        from zarr.storage import FsspecStore, MemoryStore

        storage_options = {}
        if path.startswith('s3://'):
            storage_options['anon'] = True

        store = CacheStore(
            FsspecStore.from_url(path, storage_options=storage_options),
            cache_store=MemoryStore(),
            max_size=cache_mb * 1_000_000,
        )
    else:
        store = zarr.storage.LocalStore(path)

    root = zarr.open_group(store, mode='r')
    group, ms = _find_multiscales(root)

    if ms is not None:
        datasets = ms.get('datasets', [])
    else:
        group = root
        datasets = []

    if not datasets:
        children = sorted(group.keys(), key=lambda k: (len(k), k))
        datasets = [{'path': k} for k in children]

    if num_levels is not None:
        datasets = datasets[:num_levels]

    arrays = [group[ds['path']] for ds in datasets]

    scale = None
    try:
        transforms = datasets[0].get('coordinateTransformations', [])
        for t in transforms:
            if t.get('type') == 'scale':
                scale = t['scale']
                break
    except (KeyError, IndexError):
        pass

    return arrays, scale


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Open an OME-Zarr with napari progressive loading',
    )
    parser.add_argument('path', help='Local path or remote URL to an OME-Zarr')
    parser.add_argument('--levels', type=int, default=None,
                        help='Number of multiscale levels (default: all)')
    parser.add_argument('--contrast', type=float, nargs=2, metavar=('LO', 'HI'),
                        default=None, help='Contrast limits')
    parser.add_argument('--colormap', default='gray', help='Colormap name')
    parser.add_argument('--cache-mb', type=int, default=4000,
                        help='Remote cache size in MB (default: 4000)')
    parser.add_argument('--3d', dest='threed', action='store_true',
                        help='Start in 3D display mode')
    args = parser.parse_args(argv)

    arrays, scale = open_ome_zarr(args.path, num_levels=args.levels,
                                  cache_mb=args.cache_mb)

    kwargs = {'colormap': args.colormap}
    if args.contrast is not None:
        kwargs['contrast_limits'] = tuple(args.contrast)
    if scale is not None:
        kwargs['scale'] = scale

    viewer = napari.Viewer()
    if args.threed:
        viewer.dims.ndisplay = 3

    add_progressive_loading_image(arrays, viewer=viewer, **kwargs)
    napari.run()


if __name__ == '__main__':
    main()
