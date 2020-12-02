# demonstrate how to find the vispy 3D Texture for a layer
# and how to update part of it

import numpy as np
import napari
from napari.qt.threading import thread_worker
from pathlib import Path
import time
import tifffile
from itertools import cycle

# Get list of files that simulate the camera images
# TODO: provide a public download link
files = list(Path("./splitted2").glob("*.tif"))

# Determine overall shape of a single volume
nz = max(list(map(lambda x: int(str(x.stem)[-3:]), files)))+1
single_slice = (tifffile.imread(files[0])/8).astype(np.uint8)
ny, nx = single_slice.shape 
print(f"Stack has shape (nz, ny, nx) = ({nz}, {ny}, {nx}).")

# blank volume to initialize the image layer before
# images come in from the camera worker thread
zeros = np.zeros_like(np.array(nz*[single_slice]))

# build the affine matrix for deskewing the 
# sample data set from Talley Lambert
deskew=np.eye(4)
deskew[2,0] = 4.086

@thread_worker
def cameraimage():
    """ Simulate reading slice images that are written 
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
    previous = np.zeros_like(single_slice)
    current = (tifffile.imread(next(file_iterator))/8).astype(np.uint8)
    
    for file in file_iterator:
        if count % nz == nz-1:
            # simulate some delay between
            # subsequent stacks
            time.sleep(0.5)
        else:
            # some limit on framerate
            time.sleep(0.00001)

        previous = current
        current = (tifffile.imread(file)/8).astype(np.uint8)
        count += 1
        yield count % nz, previous, current


with napari.gui_qt():
    # Initialize viewer with a blank volume
    viewer = napari.Viewer(ndisplay=3, title="livevolume")
    viewer.add_image(zeros, name="Live Acquisition", affine=deskew, scale=(3,1,1), contrast_limits=(0, 0.1))

    # The following three lines get us access to the vispy 3D texture object
    # Note that we are using some private methods inside napari, so this
    # is a bit hacky.
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[viewer.layers[0]]
    volume = vispy_layer._volume_node
    texture = volume._tex
    print(type(texture))
 
    def update_slice_vol(params):
        """ Callback that updates the texture buffer when the
        worker thread yields a new image.
        We highlight the current slice by adding a fixed offset
        value"""
        zslice, data_previous, data_current = params
        if zslice < nz-1:
            texture.set_data(
                np.array([data_previous, data_current+20]), offset=(zslice, 0, 0)
            )
        elif zslice == nz-1:
            texture.set_data(
                np.array([data_current, data_current]), offset=(zslice-1, 0, 0)
            )
        volume.update()

    
    worker = cameraimage()
    worker.yielded.connect(update_slice_vol)
    worker.start()
