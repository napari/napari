import tempfile

import numpy as np
import pytest
import zarr


@pytest.mark.parametrize('array_type', ['numpy', 'zarr', 'tensorstore'])
def test_labels_painting(MouseEvent, make_napari_viewer, array_type):
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
