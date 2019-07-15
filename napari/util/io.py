import numpy as np
from skimage import io


def read(filenames):
    """ Read image files and return an array.

    If multiple images are selected, they are stacked along the 0th axis.

    Parameters
    -------
    filenames : list
        List of filenames to be opened

    Returns
    -------
    image : array
        Array of images
    """
    images = [io.imread(filename) for filename in filenames]
    if len(images) == 1:
        image = images[0]
    else:
        image = np.stack(images)

    return image
