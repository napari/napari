from napari.utils.colormaps import simple_colormaps
import numpy as np
from napari.layers import Image


def split_axis(viewer, axis=None, keep=False):
    """Function to split the active layer into separate layers along an axis

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The viewer with the selected image
    axis : int
        The axis to split along. If None, pick the dimension with the least elements.
    keep : Boolean
        If true keep the original active layer, otherwise delete it
    """

    if not isinstance(viewer.active_layer, Image):
        print("Active layer is not an image")
        return

    data = viewer.active_layer.data
    name = viewer.active_layer.name

    cmaps = list(simple_colormaps.keys())
    display_dims = viewer.active_layer.dims.displayed
    num_dim = len(data.shape)

    orig = viewer.active_layer

    if axis is None:
        axis = np.array(data.shape).argmin()

    print(axis)
    if axis in display_dims:
        print("don't split x or y")
        return

    if num_dim < 3:
        print("not enough dimensions")
        return

    if axis >= num_dim:
        print("the image has {} dimensions".format(num_dim))
        return

    # data = np.moveaxis(data, axis, -1)
    # viewer.add_image(data, channel_axis=axis, blending='additive', name="split_" + name)
    for i in range(data.shape[axis]):
        vname = "C{:02d}_{}".format(i, name)
        print(i, vname)
        viewer.add_image(
            np.take(data, i, axis=axis),
            blending='additive',
            colormap=cmaps[i],
            name=vname,
        )

    if not keep:
        viewer.layers.remove(orig)

    print('done')


def combine_layers(viewer, rgb=False, keep=True):
    """Function to combine selected image layers in one layer

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The viewer with the selected image
    rgb : If rgb is True, add the new axis as the last dimension (-1) and
          set rgb=True in viewer.add_imageß
    keep : Boolean
        If true keep the original active layer, otherwise delete it
    """
    selected = viewer.layers.selected
    nc = len(selected)
    if nc < 2:
        print("less than two layers selected")
        return

    if rgb is True and (nc != 3 or nc != 4):  # not sure about alpha channel
        print("RGB image must have 3 or 4 channels")
        return

    shape = selected[0].shape
    for i in range(nc):
        if not isinstance(selected[i], Image):
            print("Selected layer {} is not an image".format(selected[i].name))
            return
        if i == 0:
            shape = selected[i].data.shape
        else:
            if shape != selected[i].data.shape:
                print("Layers not the same size")
                return

    new_list = list()
    for i in range(nc):
        new_list.append(selected[i].data)

    # this is assuming the y and x dimensions are the last 2
    if rgb:
        axis = -1
    else:
        axis = -3

    new_data = np.stack(new_list, axis=axis)
    viewer.add_image(new_data, rgb=rgb)

    if not keep:
        for s in selected:
            viewer.layers.remove(s)


if __name__ == '__main__':
    import napari

    with napari.gui_qt():
        v = napari.Viewer()
