import tempfile

import numpy as np
import pytest
import zarr

from napari.utils.interactions import mouse_press_callbacks


@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_painting(make_napari_viewer, array_type):
    """Check that painting labels paints on the canvas.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    viewer = make_napari_viewer()
    with tempfile.TemporaryDirectory(suffix='.zarr') as fn:
        if array_type == 'numpy':
            labels = np.zeros((20, 20), dtype=np.uint32)
        elif array_type in {'zarr', 'tensorstore'}:
            labels = zarr.open(
                fn, shape=(20, 20), dtype=np.uint32, chunks=(10, 10)
            )
        if array_type == 'tensorstore':
            ts = pytest.importorskip('tensorstore')
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': fn},
                'path': '',
                'metadata': {
                    'dtype': labels.dtype.str,
                    'order': labels.order,
                    'shape': labels.shape,
                },
            }
            labels = ts.open(spec, create=False, open=True).result()

        layer = viewer.add_labels(labels)
        layer.paint((10, 10), 1, refresh=True)
        visual = viewer.window._qt_viewer.layer_to_visual[layer]
        assert np.any(visual.node._data)


@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_painting_with_mouse(
    MouseEvent, make_napari_viewer, array_type
):
    """Check that painting labels paints on the canvas when using mouse.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    viewer = make_napari_viewer()
    with tempfile.TemporaryDirectory(suffix='.zarr') as fn:
        if array_type == 'numpy':
            labels = np.zeros((20, 20), dtype=np.uint32)
        elif array_type in {'zarr', 'tensorstore'}:
            labels = zarr.open(
                fn, shape=(20, 20), dtype=np.uint32, chunks=(10, 10)
            )
        if array_type == 'tensorstore':
            ts = pytest.importorskip('tensorstore')
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': fn},
                'path': '',
                'metadata': {
                    'dtype': labels.dtype.str,
                    'order': labels.order,
                    'shape': labels.shape,
                },
            }
            labels = ts.open(spec, create=False, open=True).result()

        layer = viewer.add_labels(labels)
        layer.mode = 'paint'
        event = MouseEvent(
            type='mouse_press',
            button=1,
            position=(0, 10, 10),
            dims_displayed=(0, 1),
        )
        visual = viewer.window._qt_viewer.layer_to_visual[layer]
        assert not np.any(visual.node._data)
        mouse_press_callbacks(layer, event)
        visual = viewer.window._qt_viewer.layer_to_visual[layer]
        assert np.any(visual.node._data)
