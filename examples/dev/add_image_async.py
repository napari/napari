# A simple driving example to test async slicing.

from skimage import data
import napari

viewer = napari.view_image(data.brain())

if __name__ == '__main__':
    napari.run()
