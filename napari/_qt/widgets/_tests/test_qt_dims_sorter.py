from napari._qt.widgets.qt_dims_sorter import QtDimsSorter


def test_dims_sorter(make_napari_viewer):
    viewer = make_napari_viewer()
    dim_sorter = QtDimsSorter(viewer)
    assert tuple(dim_sorter.axes_list) == (0, 1)

    viewer.dims.axis_labels = ('y', 'x')
    assert tuple(dim_sorter.axes_list) == ('y', 'x')

    dim_sorter.axes_list.move(1, 0)
    assert tuple(dim_sorter.axes_list) == ('x', 'y')
    assert tuple(viewer.dims.order) == (1, 0)


def test_dims_sorter_with_reordered_init(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.dims.order = (1, 0)

    dim_sorter = QtDimsSorter(viewer)
    assert tuple(dim_sorter.axes_list) == tuple(viewer.dims.order)
