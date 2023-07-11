"""
Live tiffs
==========

Loads and Displays tiffs as they get generated in the specific directory.
Trying to simulate the live display of data as it gets acquired by microscope.
This script should be run together with live_tiffs_generator.py

.. tags:: experimental
"""

import os
import sys
import time

import dask.array as da
from dask import delayed
from skimage.io.collection import alphanumeric_key
from tifffile import imread

import napari
from napari.qt import thread_worker

viewer = napari.Viewer(ndisplay=3)
# pass a directory to monitor or it will monitor current directory.
path = sys.argv[1] if len(sys.argv) > 1 else '.'
path = os.path.abspath(path)
end_of_experiment = 'final.log'


def append(delayed_image):
    """Appends the image to viewer.

    Parameters
    ----------
    delayed_image : dask.delayed function object
    """
    if delayed_image is None:
        return

    if viewer.layers:
        # layer is present, append to its data
        layer = viewer.layers[0]
        image_shape = layer.data.shape[1:]
        image_dtype = layer.data.dtype
        image = da.from_delayed(
            delayed_image, shape=image_shape, dtype=image_dtype,
        ).reshape((1, *image_shape))
        layer.data = da.concatenate((layer.data, image), axis=0)
    else:
        # first run, no layer added yet
        image = delayed_image.compute()
        image = da.from_delayed(
            delayed_image, shape=image.shape, dtype=image.dtype,
        ).reshape((1, *image.shape))
        layer = viewer.add_image(image, rendering='attenuated_mip')

    # we want to show the last file added in the viewer to do so we want to
    # put the slider at the very end. But, sometimes when user is scrolling
    # through the previous slide then it is annoying to jump to last
    # stack as it gets added. To avoid that jump we 1st check where
    # the scroll is and if its not at the last slide then don't move the slider.
    if viewer.dims.point[0] >= layer.data.shape[0] - 2:
        viewer.dims.set_point(0, layer.data.shape[0] - 1)


@thread_worker(connect={'yielded': append})
def watch_path(path):
    """Watches the path for new files and yields it once file is ready.

    Notes
    -----
    Currently, there is no proper way to know if the file has written
    entirely. So the workaround is we assume that files are generating
    serially (in most microscopes it common), and files are name in
    alphanumeric sequence We start loading the total number of minus the
    last file (`total__files - last`). In other words, once we see the new
    file in the directory, it means the file before it has completed so load
    that file. For this example, we also assume that the microscope is
    generating a `final.log` file at the end of the acquisition, this file
    is an indicator to stop monitoring the directory.

    Parameters
    ----------
    path : str
    directory to monitor and load tiffs as they start appearing.
    """
    current_files = set()
    processed_files = set()
    end_of_acquisition = False
    while not end_of_acquisition:
        files_to_process = set()
        # Get the all files in the directory at this time
        current_files = set(os.listdir(path))

        # Check if the end of acquisition has reached
        # if yes then remove it from the files_to_process set
        # and send it to display
        if end_of_experiment in current_files:
            files_to_process = current_files - processed_files
            files_to_process.remove(end_of_experiment)
            end_of_acquisition = True

        elif len(current_files):
            # get the last file from the current files based on the file names
            last_file = sorted(current_files, key=alphanumeric_key)[-1]
            current_files.remove(last_file)
            files_to_process = current_files - processed_files

        # yield every file to process as a dask.delayed function object.
        for p in sorted(files_to_process, key=alphanumeric_key):
            yield delayed(imread)(os.path.join(path, p))
        else:
            yield

        # add the files which we have yield to the processed list.
        processed_files.update(files_to_process)
        time.sleep(0.1)


worker = watch_path(path)
if __name__ == '__main__':
    napari.run()
