# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "napari[pyqt5]",
#     "zarr>=3.1.6",
#     "fsspec>=2023.10.0",
#     "aiohttp",
#     "requests",
#     "s3fs",
#     "napari-colormaps",
# ]
#
# [tool.uv.sources]
# napari = { git = "https://github.com/kephale/napari", branch = "progressive-loading-rebase" }
# napari-colormaps = { git = "https://github.com/kephale/napari-colormaps" }
# ///
"""
Progressive loading: generic OME-Zarr viewer
============================================

Open any local or remote OME-Zarr with progressive loading, or pick a
named preset::

    # Named presets — no URL needed
    uv run progressive_loading_viewer_uv.py cardiac
    uv run progressive_loading_viewer_uv.py zebrafish-em
    uv run progressive_loading_viewer_uv.py platynereis
    uv run progressive_loading_viewer_uv.py zebrahub
    uv run progressive_loading_viewer_uv.py hela
    uv run progressive_loading_viewer_uv.py covid

    # Any URL
    uv run progressive_loading_viewer_uv.py https://example.com/data.ome.zarr
    uv run progressive_loading_viewer_uv.py s3://bucket/data.zarr/path/to/group
    uv run progressive_loading_viewer_uv.py /local/path.zarr

    # List all presets
    uv run progressive_loading_viewer_uv.py --list

Options::

    --levels N        Number of multiscale levels to load (default: all)
    --contrast LO HI  Contrast limits (default: auto or preset)
    --colormap NAME   Colormap name (default: gray or preset)
    --cache-mb N      In-memory cache size in MB (default: 4000)
    --3d              Start in 3D display mode
    --rendering NAME  3D rendering mode (default: attenuated_mip)

.. tags:: experimental
"""

import argparse

import zarr

import napari

try:
    import napari_colormaps  # noqa: F401 – registers colormaps
except ImportError:
    pass

from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

# ── Presets ────────────────────────────────────────────────────────────
# Each preset defines the URL/path plus sensible defaults so users can
# just type a short name.  All fields except 'path' are optional.

PRESETS = {
    'cardiac': {
        'desc': 'Zebrafish heart FIB-SEM — 16 B voxels, 15-level pyramid (COSEM)',
        'path': 's3://janelia-cosem-datasets/jrc_zf-cardiac-1/jrc_zf-cardiac-1.zarr/recon-1/em/fibsem-uint8',
        'levels': 15,
        'contrast': (0, 255),
        'colormap': 'deep_sea_cb',
        'threed': True,
        'rendering': 'attenuated_mip',
    },
    'zebrafish-em': {
        'desc': 'Zebrafish embryo sagittal EM — 350 Gpx, 8 levels (IDR idr0053)',
        'path': 'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/4495402.zarr/',
        'levels': 8,
        'contrast': (0, 255),
        'colormap': 'phantom_blue',
        'zarr_format': 2,
    },
    'platynereis': {
        'desc': 'Platynereis whole-worm serial EM — 10 levels, anisotropic (EMBL)',
        'path': 'https://s3.embl.de/i2k-2020/platy-raw.ome.zarr',
        'levels': 10,
        'contrast': (0, 255),
        'colormap': 'bioluminescent',
        'threed': True,
        'rendering': 'attenuated_mip',
    },
    'zebrahub': {
        'desc': 'Zebrafish embryo light-sheet 4D timelapse — 1100 tp (CZ Biohub)',
        'path': 'https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS002.ome.zarr/',
        'levels': 4,
        'contrast': (0, 1000),
        'colormap': 'ocean_blue',
    },
    'hela': {
        'desc': 'HeLa cell FIB-SEM + organelle labels (COSEM)',
        'path': 's3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8',
        'levels': 6,
        'contrast': (0, 255),
        'colormap': 'polar_night',
        'threed': True,
        'rendering': 'attenuated_mip',
    },
    'covid': {
        'desc': 'SARS-CoV-2 infected cell FIB-SEM, uint16 (COSEM)',
        'path': 's3://janelia-cosem-datasets/jrc_ccl81-covid-1/jrc_ccl81-covid-1.zarr/recon-1/em/fibsem-uint16',
        'levels': 5,
        'contrast': (0, 65535),
        'colormap': 'hot_plasma',
        'threed': True,
        'rendering': 'attenuated_mip',
    },
    'mouse-kidney': {
        'desc': 'Mouse kidney FIB-SEM (COSEM / OpenOrganelle)',
        'path': 's3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.zarr/recon-1/em/fibsem-uint8',
        'levels': 5,
        'contrast': (0, 255),
        'colormap': 'earth_clay_cb',
        'threed': True,
        'rendering': 'attenuated_mip',
    },
    'idr-em-2d': {
        'desc': 'SARS-CoV-2 intestinal organoid TEM — 13 Gpx (IDR idr0083)',
        'path': 'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0083A/9822152.zarr/',
        'levels': 11,
        'contrast': (0, 65535),
        'colormap': 'dark_cobalt',
    },
    'mandelbrot': {
        'desc': 'Generative Mandelbrot set (local, no network)',
        'path': '__generative_mandelbrot__',
        'colormap': 'twilight_shifted',
        'contrast': (0, 255),
    },
    'mandelbulb': {
        'desc': 'Generative Mandelbulb (local, no network)',
        'path': '__generative_mandelbulb__',
        'colormap': 'sunset_blaze',
        'contrast': (0, 255),
        'threed': True,
        'rendering': 'attenuated_mip',
    },
}


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


def open_ome_zarr(path: str, *, num_levels: int | None = None,
                  cache_mb: int = 4000, zarr_format: int | None = None):
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

    open_kw = {'mode': 'r'}
    if zarr_format is not None:
        open_kw['zarr_format'] = zarr_format

    root = zarr.open_group(store, **open_kw)
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

    # Squeeze singleton leading dims (e.g. IDR v0.1 5D → 2D)
    if arrays and arrays[0].ndim > 3:
        shape0 = arrays[0].shape
        leading_singletons = sum(1 for s in shape0 if s == 1)
        if leading_singletons >= arrays[0].ndim - 2:
            import dask.array as da
            arrays = [da.from_zarr(a).squeeze() for a in arrays]

    scale = None
    try:
        transforms = datasets[0].get('coordinateTransformations', [])
        for t in transforms:
            if t.get('type') == 'scale':
                scale = t['scale']
                break
    except (KeyError, IndexError):
        pass

    # Trim scale to match squeezed ndim
    if scale is not None and arrays:
        arr_ndim = arrays[0].ndim if not hasattr(arrays[0], 'ndim') else arrays[0].ndim
        if len(scale) > arr_ndim:
            scale = scale[-arr_ndim:]

    return arrays, scale


def open_generative(name: str):
    from napari.experimental._progressive_loading_datasets import (
        mandelbrot_dataset,
        mandelbulb_dataset,
    )
    if name == '__generative_mandelbrot__':
        ds = mandelbrot_dataset(max_levels=14)
    else:
        ds = mandelbulb_dataset(max_levels=6)
    return ds['arrays'], None


def list_presets():
    print('\nAvailable presets:\n')
    max_name = max(len(n) for n in PRESETS)
    for name, p in PRESETS.items():
        flag = '3D' if p.get('threed') else '2D'
        print(f'  {name:<{max_name}}  [{flag}]  {p["desc"]}')
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Open an OME-Zarr with napari progressive loading',
    )
    parser.add_argument('path', nargs='?', default=None,
                        help='Preset name, local path, or remote URL')
    parser.add_argument('--list', action='store_true',
                        help='List available presets and exit')
    parser.add_argument('--levels', type=int, default=None,
                        help='Number of multiscale levels (default: all)')
    parser.add_argument('--contrast', type=float, nargs=2, metavar=('LO', 'HI'),
                        default=None, help='Contrast limits')
    parser.add_argument('--colormap', default=None, help='Colormap name')
    parser.add_argument('--cache-mb', type=int, default=4000,
                        help='Remote cache size in MB (default: 4000)')
    parser.add_argument('--3d', dest='threed', action='store_true',
                        default=None, help='Start in 3D display mode')
    parser.add_argument('--rendering', default=None,
                        help='3D rendering mode (default: attenuated_mip)')
    args = parser.parse_args(argv)

    if args.list:
        list_presets()
        return

    if args.path is None:
        parser.print_help()
        print('\nUse --list to see available presets.')
        return

    # Resolve preset or raw path
    preset = PRESETS.get(args.path, {})
    data_path = preset.get('path', args.path)

    # Merge: CLI flags override preset defaults
    num_levels = args.levels or preset.get('levels')
    contrast = args.contrast or preset.get('contrast')
    colormap = args.colormap or preset.get('colormap', 'gray')
    threed = args.threed if args.threed is not None else preset.get('threed', False)
    rendering = args.rendering or preset.get('rendering', 'attenuated_mip')
    zarr_format = preset.get('zarr_format')

    # Open data
    if data_path.startswith('__generative_'):
        arrays, scale = open_generative(data_path)
    else:
        print(f'Opening {data_path} ...')
        arrays, scale = open_ome_zarr(data_path, num_levels=num_levels,
                                      cache_mb=args.cache_mb,
                                      zarr_format=zarr_format)

    print(f'  {len(arrays)} levels, level 0: shape={arrays[0].shape} '
          f'dtype={arrays[0].dtype}')

    kwargs = {'colormap': colormap}
    if contrast is not None:
        kwargs['contrast_limits'] = tuple(contrast)
    if scale is not None:
        kwargs['scale'] = scale
    if threed:
        kwargs['rendering'] = rendering

    viewer = napari.Viewer()
    if threed:
        viewer.dims.ndisplay = 3

    add_progressive_loading_image(arrays, viewer=viewer, **kwargs)
    napari.run()


if __name__ == '__main__':
    main()
