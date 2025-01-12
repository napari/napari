import dask.array as da
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider

from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
    get_layer_name_for_scale,
    initialize_multiscale_virtual_data,
)


def open_zebrahub():
    url = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS002.ome.zarr/"

    # read the image data
    parse_url(url, mode="r").store

    reader = Reader(parse_url(url))
    # nodes may include images, labels etc
    nodes = list(reader())
    # first node will be the image pixel data
    image_node = nodes[0]

    dask_data = image_node.data

    return dask_data


import toolz as tz


@tz.curry
def update_timepoint(timepoint: int, timeseries_img=None, viewer=None):
    # TODO reconnect the new data to the event listeners

    print(f"Updating timepoint {timepoint}")

    if timeseries_img is None or viewer is None:
        print("timeseries_img or viewer is None. Cannot update timepoint.")
        return

    # Extract the image data for the given timepoint and adjust the dimensions
    multiscale_img = [da.transpose(da.squeeze(img[timepoint, 0, :, :, :]), (2, 1, 0)) for img in timeseries_img]
    print(f"multiscale img: {multiscale_img} shapes {[img.shape for img in timeseries_img]}")

    # Initialize multiscale virtual data for the new timepoint
    multiscale_vdata = initialize_multiscale_virtual_data(multiscale_img, viewer, ndisplay=3)

    if multiscale_vdata is not None:

        # Update the image layer with the new timepoint data
        for scale, img in enumerate(multiscale_vdata._data):
            scale_name = get_layer_name_for_scale(scale)
            if scale_name in viewer.layers:
                viewer.layers[scale_name].data = img
            else:
                print(f"Layer {scale_name} not found in viewer.")

if __name__ == "__main__":
    import napari

    viewer = napari.Viewer(ndisplay=3)

    # multiscale_img = open_zebrahub()
    timeseries_img = open_zebrahub()
    multiscale_img = [
        da.transpose(da.squeeze(img[0, 0, :, :, :]), (2, 1, 0))
        for img in timeseries_img
    ]

    # Create a slider to control the timepoint
    slider = QSlider()
    slider.setOrientation(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(
        timeseries_img[-1].shape[0] - 1
    )  # Set the maximum value to the number of timepoints
    slider.setValue(0)

    print(multiscale_img[0])

    add_progressive_loading_image(
        multiscale_img,
        viewer=viewer,
        contrast_limits=[0, 255],
        colormap='twilight_shifted',
        ndisplay=3,
    )

    # Connect the slider value change to the update_timepoint function
    slider.valueChanged.connect(
        update_timepoint(timeseries_img=timeseries_img, viewer=viewer)
    )

    # Add the slider to the viewer layout
    viewer.window.add_dock_widget(slider, area='left')

    # Initial loading of the image at timepoint 550
    update_timepoint(0, timeseries_img=timeseries_img, viewer=viewer)

    viewer.axes.visible = True

    napari.run()
