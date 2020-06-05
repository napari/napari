from napari.utils.colormaps import simple_colormaps
import numpy as np
from napari.layers import Image


def stack_to_images(stack, axis):
    """Function to split the active layer into separate layers along an axis

    Parameters
    ----------
    stack : napari.layers.Image
        The image stack to be split into a list of image layers
    axis : int
        The axis to split along.

    Returns
    -------
    list
        List of images
    """

    if not isinstance(stack, Image):
        print("Active layer is not an image")
        return None

    data = stack.data
    name = stack.name

    cmaps = list(simple_colormaps.keys())
    num_dim = len(data.shape)

    if num_dim < 3:
        print("not enough dimensions")
        return None

    if axis >= num_dim:
        print("the image has {} dimensions".format(num_dim))
        return None

    imagelist = list()
    for i in range(data.shape[axis]):
        layer_name = "{:02d}_{}".format(i, name)

        try:
            color = cmaps[i]
        except IndexError:
            color = 'gray'

        image = Image(
            np.take(data, i, axis=axis),
            blending='additive',
            colormap=color,
            name=layer_name,
        )

        imagelist.append(image)

    return imagelist


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
