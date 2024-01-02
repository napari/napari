import numpy as np

from napari_builtins._ndims_balls import (
    labeled_particles2d,
    labeled_particles3d,
)


def test_labeled_particles2d():
    img, labels, points = labeled_particles2d()
    assert img[0].ndim == 2
    assert labels[0].ndim == 2
    assert "seed" in img[1]["metadata"]
    assert "seed" in labels[1]["metadata"]
    assert "seed" in points[1]["metadata"]
    assert img[2] == "image"
    assert labels[2] == "labels"
    assert points[2] == "points"

    assert np.all(img[0][labels[0] > 0] > 0)


def test_labeled_particles3d():
    img, labels, points = labeled_particles3d()
    assert img[0].ndim == 3
    assert labels[0].ndim == 3
    assert "seed" in img[1]["metadata"]
    assert "seed" in labels[1]["metadata"]
    assert "seed" in points[1]["metadata"]
    assert img[2] == "image"
    assert labels[2] == "labels"
    assert points[2] == "points"

    assert np.all(img[0][labels[0] > 0] > 0)
