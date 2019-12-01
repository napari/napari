import numpy as np
from skimage.data import binary_blobs
from imageio import imwrite
from scipy import ndimage as ndi
from tempfile import NamedTemporaryFile
from napari.components import ViewerModel


def test_png_labels():
    viewer = ViewerModel()
    blobs = binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
    labeled = ndi.label(blobs)[0].astype(np.uint8)
    with NamedTemporaryFile(suffix='.png') as fout:
        imwrite(fout, labeled, format='png')
        viewer.add_labels(path=fout.name)
        assert len(viewer.layers) == 1
        assert np.all(labeled == viewer.layers[0].data)


def test_tiff_labels():
    viewer = ViewerModel()
    blobs = binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
    labeled = ndi.label(blobs)[0].astype(np.uint8)
    with NamedTemporaryFile(suffix='.tiff') as fout:
        imwrite(fout, labeled, format='tiff')
        viewer.add_labels(path=fout.name)
        assert len(viewer.layers) == 1
        assert np.all(labeled == viewer.layers[0].data)
