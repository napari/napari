import pytest
import numpy as np
from skimage.data import binary_blobs
from imageio import imwrite
from scipy import ndimage as ndi
from napari.components import ViewerModel
from napari.utils.temporary_file import temporary_file


@pytest.mark.parametrize('suffix', ['.png', '.tiff'])
def test_open_labels(suffix):
    viewer = ViewerModel()
    blobs = binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
    labeled = ndi.label(blobs)[0].astype(np.uint8)
    with temporary_file(suffix) as fout:
        imwrite(fout, labeled, format=suffix)
        viewer.add_labels(path=fout)
        assert len(viewer.layers) == 1
        assert np.all(labeled == viewer.layers[0].data)
