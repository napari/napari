import numpy as np
import pytest
import zarr
from qtpy.QtCore import QCoreApplication

from napari._tests.utils import skip_local_popups
from napari.utils.interactions import mouse_press_callbacks


def make_labels_layer(array_type, path, shape):
    """Make a labels layer, either NumPy, zarr, or tensorstore."""
    chunks = tuple(s // 2 for s in shape)
    if array_type == 'numpy':
        labels = np.zeros(shape, dtype=np.uint32)
    elif array_type in {'zarr', 'tensorstore'}:
        labels = zarr.open(path, shape=shape, dtype=np.uint32, chunks=chunks)
    if array_type == 'tensorstore':
        ts = pytest.importorskip('tensorstore')
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(path)},
            'path': '',
            'metadata': {
                'dtype': labels.dtype.str,
                'order': labels.order,
                'shape': labels.shape,
            },
        }
        labels = ts.open(spec, create=False, open=True).result()

    return labels


@skip_local_popups
@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_painting(make_napari_viewer, array_type, tmp_path):
    """Check that painting labels paints on the canvas.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    viewer = make_napari_viewer(show=True)
    labels = make_labels_layer(array_type, tmp_path, shape=(20, 20))
    layer = viewer.add_labels(labels)
    QCoreApplication.instance().processEvents()
    layer.paint((10, 10), 1, refresh=True)
    visual = viewer.window._qt_viewer.layer_to_visual[layer]
    assert np.any(visual.node._data)


@skip_local_popups
@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_fill_slice(make_napari_viewer, array_type, tmp_path):
    """Check that painting labels paints only on current slice.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    viewer = make_napari_viewer(show=True)
    labels = make_labels_layer(array_type, tmp_path, shape=(3, 20, 20))
    labels[0, :, :] = 1
    labels[1, 10, 10] = 1
    labels[2, :, :] = 1
    layer = viewer.add_labels(labels)
    layer.n_edit_dimensions = 3
    QCoreApplication.instance().processEvents()
    layer.fill((1, 10, 10), 13, refresh=True)
    visual = viewer.window._qt_viewer.layer_to_visual[layer]
    assert np.sum(visual.node._data) == 13


@skip_local_popups
@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_painting_with_mouse(
    MouseEvent, make_napari_viewer, array_type, tmp_path
):
    """Check that painting labels paints on the canvas when using mouse.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    viewer = make_napari_viewer(show=True)
    labels = make_labels_layer(array_type, tmp_path, shape=(20, 20))

    layer = viewer.add_labels(labels)
    QCoreApplication.instance().processEvents()

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
    assert np.any(visual.node._data)
