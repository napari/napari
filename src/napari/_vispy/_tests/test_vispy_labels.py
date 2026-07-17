import numpy as np
import pytest

import napari._vispy.layers.base as vispy_base
import napari._vispy.layers.labels as vispy_labels
from napari._vispy.layers.labels import (
    DirectLabelVispyColormap,
    LabelVispyColormap,
    VispyLabelsLayer,
    build_textures_from_dict,
)
from napari._vispy.utils.qt_font import FontInfo
from napari.layers import Labels
from napari.settings import get_settings
from napari.utils.colormaps import DirectLabelColormap


@pytest.fixture
def _mock_max_texture_sizes(monkeypatch):
    max_texture_sizes = (16384, 2048)
    monkeypatch.setattr(
        vispy_base, 'get_max_texture_sizes', lambda: max_texture_sizes
    )
    monkeypatch.setattr(
        vispy_labels, 'get_max_texture_sizes', lambda: max_texture_sizes
    )


def test_build_textures_from_dict():
    values = build_textures_from_dict(
        {0: (0, 0, 0, 0), 1: (1, 1, 1, 1), 2: (2, 2, 2, 2)},
        max_size=10,
    )
    assert values.shape == (3, 1, 4)
    assert np.array_equiv(values[1], (1, 1, 1, 1))
    assert np.array_equiv(values[2], (2, 2, 2, 2))


def test_build_textures_from_dict_exc():
    with pytest.raises(ValueError, match='Cannot create a 2D texture'):
        build_textures_from_dict(
            {0: (0, 0, 0, 0), 1: (1, 1, 1, 1), 2: (2, 2, 2, 2)},
            max_size=1,
        )


@pytest.mark.usefixtures('_fresh_settings', '_mock_max_texture_sizes')
def test_colormap_rebuilt_when_slice_dtypes_change():
    get_settings().experimental.async_ = True
    data = np.zeros((8, 8), dtype=np.uint32)
    data[0, 0] = 100_000
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
