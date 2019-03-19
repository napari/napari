import numpy
from skimage.transform import rescale
from PIL import Image
import requests
from io import BytesIO

__bluemarble_url = """https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/
    73751/world.topo.bathy.200407.3x5400x2700.jpg"""
__bluemarble_large_url = """https://eoimages.gsfc.nasa.gov/images/imagerecords/
    57000/57730/land_ocean_ice_8192.png"""
__bluemarble_verylarge_url = """https://eoimages.gsfc.nasa.gov/images/
    imagerecords/73000/73751/world.topo.bathy.200407.3x21600x10800.jpg"""
__bluemarble_large_githuburl = """https://github.com/Napari/napari-data/blob/
    master/XYrgb/world.topo.bathy.200407.3x16384x8192.jpg?raw=true"""


def load_bluemarble_image(large=False):
    if large:
        return load_image_array_from_url(__bluemarble_large_githuburl)
    else:
        return load_image_array_from_url(__bluemarble_url)


def load_image_array_from_url(url, scale=1.0):
    print("Downloading from: %s with scale: %f" % (url, scale))
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    array = numpy.array(img)
    print(array.shape)
    if scale != 1.0:
        array_rescaled = rescale(array, scale)
        print(array_rescaled.shape)
        return array_rescaled
    else:
        return array
