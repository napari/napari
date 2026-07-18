import numpy as np
import pytest
import zarr
from qtpy.QtCore import QCoreApplication

import napari._vispy.layers.base as vispy_base
import napari._vispy.layers.labels as vispy_labels
from napari._tests.utils import skip_local_popups, skip_on_win_ci
from napari._vispy.layers.labels import (
    DirectLabelVispyColormap,
    LabelVispyColormap,
    VispyLabelsLayer,
)
from napari._vispy.utils.qt_font import FontInfo
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.layers import Labels
from napari.layers.labels._labels_constants import IsoCategoricalGradientMode
from napari.settings import get_settings
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.interactions import mouse_press_callbacks


@pytest.fixture
def _mock_max_texture_sizes(monkeypatch):
    max_texture_sizes = (16384, 2048)
    monkeypatch.setattr(
        vispy_base, 'get_max_texture_sizes', lambda: max_texture_sizes
    )
    monkeypatch.setattr(
        vispy_labels, 'get_max_texture_sizes', lambda: max_texture_sizes
    )


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


@pytest.mark.usefixtures('_fresh_settings', '_mock_max_texture_sizes')
def test_colormap_rebuilt_when_slice_dtypes_change():
    get_settings().experimental.async_ = True
    data = np.zeros((8, 8), dtype=np.uint32)
    data[0, 0] = 42
    layer = Labels(data)
    visual = VispyLabelsLayer(layer, font_info=FontInfo())

    try:
        layer.colormap = DirectLabelColormap(
            color_dict={
                None: [0, 0, 0, 0],
                0: [0, 0, 0, 0],
                1: [0, 0.25, 1, 1],
                100_000: [1, 0, 0, 1],
            }
        )
        assert layer._slice.empty
        # The empty placeholder uses uint8 for both raw and view, so the
        # colormap is initially configured for that pair. Retaining this state
        # for the real (uint32, uint8) slice would produce stale label colors.
        # _on_data_change detects this transition and rebuilds the colormap.
        assert visual._colormap_dtypes == (np.dtype('uint8'),) * 2
        assert isinstance(visual.node.cmap, LabelVispyColormap)

        layer.set_view_slice()
        layer.events.set_data()

        # set_data must rebuild the colormap instead of leaving the
        # LabelVispyColormap configured for the placeholder.
        assert visual._colormap_dtypes == (
            np.dtype('uint32'),
            np.dtype('uint8'),
        )
        assert isinstance(visual.node.cmap, DirectLabelVispyColormap)
    finally:
        visual.close()


@pytest.mark.usefixtures('_mock_max_texture_sizes')
def test_colormap_not_rebuilt_when_slice_dtypes_are_unchanged():
    layer = Labels(np.zeros((8, 8), dtype=np.uint32))
    visual = VispyLabelsLayer(layer, font_info=FontInfo())

    try:
        colormap = visual.node.cmap

        layer.events.set_data()

        assert visual.node.cmap is colormap
    finally:
        visual.close()


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
