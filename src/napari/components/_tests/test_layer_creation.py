import numpy as np
import pytest

from napari._app_model.actions._file import _new_annotation_layer
from napari.components import ViewerModel
from napari.layers import Labels, Points, Shapes


@pytest.mark.parametrize('layer_class', [Labels, Points, Shapes])
@pytest.mark.parametrize('ndim', [2, 3])
def test_trailing_annotation_dimensions(layer_class, ndim):
    viewer = ViewerModel()
    source = viewer.add_image(
        np.zeros((5, 10, 20, 30)),
        scale=(7, 4, 2, 1),
        translate=(11, 12, 13, 14),
        axis_labels=('t', 'z', 'y', 'x'),
        units=('s', 'um', 'um', 'um'),
    )
    _new_annotation_layer(viewer, layer_class, ndim)
    layer = viewer.layers[-1]
    assert layer.ndim == ndim
    np.testing.assert_array_equal(layer.scale, source.scale[-ndim:])
    np.testing.assert_array_equal(layer.translate, source.translate[-ndim:])
    assert layer.axis_labels == source.axis_labels[-ndim:]
    assert layer.units == source.units[-ndim:]
    if layer_class is Labels:
        assert layer.data.shape == source.data.shape[-ndim:]
        assert not layer.data.any()
    else:
        assert len(layer.data) == 0


@pytest.mark.parametrize('layer_class', [Labels, Points, Shapes])
def test_trailing_annotation_transforms(layer_class):
    viewer = ViewerModel()
    rotate = np.eye(4)
    rotate[-2:, -2:] = [[0, -1], [1, 0]]
    shear = np.eye(4)
    shear[-2, -1] = 0.3
    affine = np.eye(5)
    affine[1, 2] = 0.2
    affine[:4, -1] = [1, 2, 3, 4]
    source = viewer.add_image(
        np.zeros((5, 10, 20, 30)),
        scale=(7, 4, 2, 1),
        translate=(11, 12, 13, 14),
        rotate=rotate,
        shear=shear,
        affine=affine,
    )
    _new_annotation_layer(viewer, layer_class, 3)
    layer = viewer.layers[-1]
    np.testing.assert_allclose(
        layer.data_to_world((2, 3, 4)), source.data_to_world((0, 2, 3, 4))[1:]
    )
    np.testing.assert_allclose(layer.rotate, source.rotate[1:, 1:])
    np.testing.assert_allclose(
        layer.shear,
        source._transforms['data2physical'].set_slice([1, 2, 3]).shear,
    )


@pytest.mark.parametrize('layer_class', [Labels, Points, Shapes])
def test_small_retained_affine_component(layer_class):
    viewer = ViewerModel()
    affine = np.eye(5)
    affine[2, 3] = 1e-9
    source = viewer.add_image(np.zeros((5, 10, 20, 30)), affine=affine)
    _new_annotation_layer(viewer, layer_class, 3)
    np.testing.assert_allclose(
        viewer.layers[-1].data_to_world((0, 0, 1e12)),
        source.data_to_world((0, 0, 0, 1e12))[1:],
    )


@pytest.mark.parametrize('layer_class', [Labels, Points, Shapes])
@pytest.mark.parametrize(
    ('transform', 'indices'),
    [
        ('affine', (0, 1)),
        ('affine', (1, 0)),
        ('rotate', (0, 1)),
        ('rotate', (1, 0)),
        ('shear', (0, 1)),
    ],
)
@pytest.mark.parametrize('coupling', [0.5, 1e-13])
def test_coupled_axes_rejected(layer_class, transform, indices, coupling):
    viewer = ViewerModel()
    matrix = np.eye(5 if transform == 'affine' else 4)
    matrix[indices] = coupling
    viewer.add_image(np.zeros((5, 10, 20, 30)), **{transform: matrix})
    with pytest.raises(ValueError, match='coupled'):
        _new_annotation_layer(viewer, layer_class, 3)
    assert len(viewer.layers) == 1


@pytest.mark.parametrize('layer_class', [Labels, Points, Shapes])
def test_multiple_selection_trailing_extent(layer_class):
    viewer = ViewerModel()
    first = viewer.add_image(np.zeros((5, 10, 20, 30)), scale=(2, 3, 4, 5))
    second = viewer.add_image(np.zeros((3, 8, 12, 16)), translate=(1, 2, 3, 4))
    viewer.layers.selection = {first, second}
    extent = viewer.layers.get_extent(viewer.layers.selection)
    _new_annotation_layer(viewer, layer_class, 3)
    layer = viewer.layers[-1]
    np.testing.assert_allclose(layer.scale, extent.step[-3:])
    if layer_class is Labels:
        shape = (
            np.round((extent.world[1] - extent.world[0]) / extent.step).astype(
                int
            )
            + 1
        )
        assert layer.data.shape == tuple(shape[-3:])
        np.testing.assert_allclose(layer.translate, extent.world[0, -3:])


def test_rgb_trailing_labels():
    viewer = ViewerModel()
    viewer.add_image(np.zeros((5, 10, 20, 30, 3)), rgb=True)
    viewer._new_labels(ndim=3)
    assert viewer.layers[-1].data.shape == (10, 20, 30)


@pytest.mark.parametrize('layer_class', [Labels, Points, Shapes])
def test_empty_viewer_annotation(layer_class):
    viewer = ViewerModel()
    _new_annotation_layer(viewer, layer_class, 2)
    assert viewer.layers[-1].ndim == 2
    if layer_class is Labels:
        assert viewer.layers[-1].data.shape == (512, 512)


def test_multiscale_trailing_labels():
    viewer = ViewerModel()
    viewer.add_image(
        [np.zeros((5, 10, 20, 30)), np.zeros((5, 5, 10, 15))],
        multiscale=True,
    )
    viewer._new_labels(ndim=3)
    assert viewer.layers[-1].data.shape == (10, 20, 30)


def test_non_scalar_reference_labels_extent():
    viewer = ViewerModel()
    source = viewer.add_points(
        [[0, 1, 2, 3], [4, 9, 19, 29]],
        scale=(7, 4, 2, 1),
        translate=(11, 12, 13, 14),
        axis_labels=('t', 'z', 'y', 'x'),
    )
    extent = viewer.layers.get_extent(viewer.layers.selection)
    viewer._new_labels(ndim=3)
    labels = viewer.layers[-1]
    shape = (
        np.round((extent.world[1] - extent.world[0]) / extent.step).astype(int)
        + 1
    )
    assert labels.data.shape == tuple(shape[-3:])
    np.testing.assert_allclose(labels.translate, extent.world[0, -3:])
    np.testing.assert_allclose(labels.scale, extent.step[-3:])
    assert labels.axis_labels == source.axis_labels[-3:]


def test_labels_visible_and_editable_across_time():
    viewer = ViewerModel()
    viewer.add_image(np.zeros((5, 10, 20, 30)))
    viewer._new_labels(ndim=3)
    labels = viewer.layers[-1]
    labels.mode = 'paint'
    labels.brush_size = 1
    for time in (0, 4):
        viewer.dims.set_point(0, time)
        viewer.dims.set_point(1, 2)
        labels.paint((2, 3, 4), time + 1)
        np.testing.assert_array_equal(labels._data_view, labels.data[2])
        assert labels._data_view[3, 4] == time + 1
        assert labels.visible
        assert labels.editable
    viewer.dims.set_point(0, 0)
    assert labels._data_view[3, 4] == 5
