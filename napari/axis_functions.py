from napari.utils.colormaps import simple_colormaps
import numpy as np


def split_axis(viewer, axis, keep=False):
    data = viewer.active_layer.data
    name = viewer.active_layer.name

    cmaps = list(simple_colormaps.keys())
    display_dims = viewer.active_layer.dims.displayed
    num_dim = len(data.shape)

    orig = viewer.active_layer

    if axis in display_dims:
        print("don't split x or y")
        return

    if num_dim < 3:
        print("not enough dimensions")
        return

    if axis >= num_dim:
        print("the image has {} dimensions".format(num_dim))
        return

    data = np.moveaxis(data, axis, -1)
    # viewer.add_image(data, channel_axis=axis, blending='additive', name="split_" + name)
    for i in range(data.shape[-1]):
        vname = "C{:02d}_{}".format(i, name)
        print(i, vname)
        viewer.add_image(
            data[..., i], blending='additive', colormap=cmaps[i], name=vname
        )

    if not keep:
        viewer.layers.remove(orig)

    print('done')


def combine_layers(viewer, keep=True):
    selected = viewer.layers.selected
    nc = len(selected)
    if nc < 2:
        print("less than two layers selected")
        return
    shape = selected[0].shape
    for i in range(nc - 1):
        if shape != selected[i + 1].shape:
            print("Layers not the same size")
            return

    new_data = np.zeros((*shape, nc), dtype=np.float32)
    new_data = np.moveaxis(new_data, -1, -3)
    print(new_data.shape)
    for i in range(nc):
        new_data[..., i, :, :] = selected[i].data
        # new_data[..., i] = selected[i].data

    viewer.add_image(new_data, blending='additive', rgb=False)
    # viewer.active_layer.dims
