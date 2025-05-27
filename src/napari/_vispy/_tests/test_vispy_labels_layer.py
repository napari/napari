import numpy as np
import pytest
import zarr
from qtpy.QtCore import QCoreApplication

from napari._tests.utils import skip_local_popups, skip_on_win_ci
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.layers.labels._labels_constants import IsoCategoricalGradientMode
from napari.utils.interactions import mouse_press_callbacks


def make_labels_layer(array_type, shape):
    """Make a labels layer, either NumPy, zarr, or tensorstore."""
    chunks = tuple(s // 2 for s in shape)
    if array_type == 'numpy':
        labels = np.zeros(shape, dtype=np.uint32)
    elif array_type == 'zarr':
        labels = zarr.zeros(shape=shape, dtype=np.uint32, chunks=chunks)
    elif array_type == 'tensorstore':
        ts = pytest.importorskip('tensorstore')
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'memory'},
            'metadata': {'chunks': chunks},
        }
        labels = ts.open(
            spec, create=True, dtype='uint32', shape=shape
        ).result()
    else:
        pytest.fail("array_type must be 'numpy', 'zarr', or 'tensorstore'")

    return labels


@skip_local_popups
@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_painting(qtbot, array_type, qt_viewer):
    """Check that painting labels paints on the canvas.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    viewer = qt_viewer.viewer
    labels = make_labels_layer(array_type, shape=(20, 20))
    layer = viewer.add_labels(labels)
    QCoreApplication.instance().processEvents()
    layer.paint((10, 10), 1, refresh=True)
    visual = qt_viewer.layer_to_visual[layer]
    assert np.any(visual.node._data)


@skip_local_popups
@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_fill_slice(qtbot, array_type, qt_viewer):
    """Check that painting labels paints only on current slice.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    labels = make_labels_layer(array_type, shape=(3, 20, 20))
    labels[0, :, :] = 1
    labels[1, 10, 10] = 1
    labels[2, :, :] = 1

    viewer = qt_viewer.viewer
    layer = viewer.add_labels(labels)
    layer.n_edit_dimensions = 3
    QCoreApplication.instance().processEvents()
    layer.fill((1, 10, 10), 13, refresh=True)
    visual = qt_viewer.layer_to_visual[layer]
    assert np.sum(visual.node._data) == 13


@skip_local_popups
@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_painting_with_mouse(MouseEvent, qtbot, array_type, qt_viewer):
    """Check that painting labels paints on the canvas when using mouse.

    This should work regardless of array type. See:
    https://github.com/napari/napari/issues/6079
    """
    labels = make_labels_layer(array_type, shape=(20, 20))

    viewer = qt_viewer.viewer
    layer = viewer.add_labels(labels)
    QCoreApplication.instance().processEvents()

    layer.mode = 'paint'
    event = MouseEvent(
        type='mouse_press',
        button=1,
        position=(0, 10, 10),
        dims_displayed=(0, 1),
    )
    visual = qt_viewer.layer_to_visual[layer]
    assert not np.any(visual.node._data)
    mouse_press_callbacks(layer, event)
    assert np.any(visual.node._data)


@skip_local_popups
@skip_on_win_ci
def test_labels_iso_gradient_modes(qtbot, qt_viewer):
    """Check that we can set `iso_gradient_mode` with `iso_categorical` rendering (test shader)."""
    # NOTE: this test currently segfaults on Windows CI, but confirmed working locally
    # because it's a segfault, we have to skip instead of xfail
    qt_viewer.show()
    viewer = qt_viewer.viewer

    labels = make_labels_layer('numpy', shape=(32, 32, 32))
    labels[14:18, 14:18, 14:18] = 1
    layer = viewer.add_labels(labels)
    visual = qt_viewer.layer_to_visual[layer]

    viewer.dims.ndisplay = 3
    QCoreApplication.instance().processEvents()
    assert layer.rendering == 'iso_categorical'
    assert isinstance(visual.node, VolumeNode)

    layer.iso_gradient_mode = IsoCategoricalGradientMode.SMOOTH
    QCoreApplication.instance().processEvents()
    assert layer.iso_gradient_mode == 'smooth'
    assert visual.node.iso_gradient_mode == 'smooth'

    layer.iso_gradient_mode = IsoCategoricalGradientMode.FAST
    QCoreApplication.instance().processEvents()
    assert layer.iso_gradient_mode == 'fast'
    assert visual.node.iso_gradient_mode == 'fast'
