from skimage import data
from skimage.color import rgb2gray

from napari import Viewer
from napari.util import app_context


# THIS WILL NOT WORK UNTIL THERE IS SEPARATION BETWEEN MODEL AND VIEW!

def test_dims_and_ranges():

    viewer  = Viewer()

    astronaut = rgb2gray(data.astronaut())
    coins = rgb2gray(data.coins())

    viewer.add_image(astronaut)
    #viewer.add_image(coins)

    assert viewer.dims.num_dimensions == 2

    assert viewer._calc_layers_num_dims() == 2

    assert viewer._calc_layers_ranges() == (12,12)
