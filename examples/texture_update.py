# demonstrate how to find the vispy 3D Texture for a layer
# and how to update part of it

import numpy as np
import napari
from napari.qt.threading import thread_worker
from pathlib import Path
import time
import tifffile
from itertools import cycle

# The texture slice of the incoming camera image
# is highlighted by adding a fixed offset to the
# grevalues. Adjust this offset to find something
# you find visually pleasing.

highlight_offset = 30

# Get list of files that simulate the camera images
# TODO: provide a public download link
img_path = Path("./lls_2ch_uint8_")
channels = ("ch0", "ch1")
_tmp = [list(img_path.glob(f"*{ch}*.tif")) for ch in channels]
files = np.array(_tmp).T

# determine volume shape from files
nz = max(list(map(lambda x: int(str(x.stem)[-3:]), files[:, 0]))) + 1
single_slice = tifffile.imread(files[0, 0])
ny, nx = single_slice.shape
print(f'Stack has shape (nz, ny, nx) = ({nz}, {ny}, {nx}).')


# blank volume to initialize the image layer before
# images come in from the camera worker thread
blank = np.zeros_like(np.array(nz * [single_slice]))

# build the affine matrix for deskewing the
# sample data set from Talley Lambert
deskew = np.eye(4)
deskew[2, 0] = 4.086


@thread_worker
def camera_simulator():
    """Simulate reading slice images that are written
    to disk by some camera acquisition software.
    Here, we just cycle through a list of images,
    each image representing one slice of a stack.

    In practice you may want to implement some
    kind of watchdog here, that monitors a folder
    for incoming images and returns them.

    Returns:

    (zslice, previous, current)

    zslice: int, current zslice number
    previous: np.ndarray, previous image
    current: np.ndarray, current image
    """

    count = -1

    file_iterator = cycle(files)

    previous = [np.zeros_like(single_slice)] * files.shape[1]
    current = [tifffile.imread(f) for f in next(file_iterator)]

    for chfiles in file_iterator:
        if count % nz == nz - 1:
            # simulate some delay between
            # subsequent stacks
            time.sleep(0.5)
        else:
            # some limit on framerate
            time.sleep(0.00001)

        previous = current
        current = [tifffile.imread(f) for f in chfiles]
        count += 1
        yield count % nz, previous, current


with napari.gui_qt():
    # Initialize viewer with a blank volume
    viewer = napari.Viewer(
        ndisplay=3, title="Live volume acquisition visualization"
    )
    spindle_l = viewer.add_image(
        blank,
        name="spindle",
        affine=deskew,
        scale=(3, 1, 1),
        colormap='green',
        blending='additive',
    )
    dna_l = viewer.add_image(
        blank,
        name="dna",
        affine=deskew,
        scale=(3, 1, 1),
        colormap='magenta',
        blending='additive',
    )

    vispy_dna_layer = viewer.window.qt_viewer.layer_to_visual[dna_l]
    vispy_spindle_layer = viewer.window.qt_viewer.layer_to_visual[spindle_l]
    volumes = [
        l._layer_node.get_node(3)
        for l in (vispy_dna_layer, vispy_spindle_layer)
    ]

    def update_slice_vol(params):
        """Callback that updates the texture buffers when the
        worker thread yields new images for the channels.
        The current slice is highlighted by adding a fixed offset."""
        zslice, data_previous, data_current = params

        for previous, current, volume in zip(
            data_previous, data_current, volumes
        ):
            texture = volume._tex
            if zslice < nz - 1:
                texture.set_data(
                    np.array([previous, current + highlight_offset]),
                    offset=(zslice, 0, 0),
                )
            elif zslice == nz - 1:
                texture.set_data(
                    np.array([current, current]), offset=(zslice - 1, 0, 0)
                )
            volume.update()

    worker = camera_simulator()
    worker.yielded.connect(update_slice_vol)
    worker.start()
