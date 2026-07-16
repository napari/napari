import numpy as np
import pytest
from imageio import imwrite
from scipy import ndimage as ndi
from skimage.data import binary_blobs

from napari.components import ViewerModel
from napari.layers import Labels


@pytest.mark.parametrize('suffix', ['.png', '.tiff'])
def test_open_labels(builtins, suffix, tmp_path):
    viewer = ViewerModel()
    blobs = binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
    labeled = ndi.label(blobs)[0].astype(np.uint8)

    fout = str(tmp_path / f'test{suffix}')
    imwrite(fout, labeled, format=suffix)
    viewer.open(fout, layer_type='labels')
    assert len(viewer.layers) == 1
    np.testing.assert_array_equal(labeled, viewer.layers[0].data)
    assert isinstance(viewer.layers[0], Labels)
