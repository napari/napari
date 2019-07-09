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


def load_numpy_array(path):
    """ Read npy or npz file and return an array volume


    Parameters
    -------
    path
        Path to npy file

    Returns
    -------
    volume : array
        3D Array of images
    """
    if path.endswith(".npy"):
        loaded_array = np.load(path)
    elif path.endswith(".npz"):
        with np.load(path) as array:
            loaded_array = array["data"]
    else:
        print("Not a valid path")
        raise AssertionError
    if loaded_array.dtype == np.bool:
        loaded_array = loaded_array.astype(np.uint8) * 255
    return loaded_array
