from skimage import filters

import napari


def gaussian(layer: napari.layers.Image) -> napari.types.LayerDataTuple:
    data = filters.gaussian(layer.data, preserve_range=True)
    name = layer.name + '_gaussian'
    dct = {**layer.as_layer_data_tuple()[1], 'name': name}
    return (data, dct, 'image')


def sobel(layer: napari.layers.Image) -> napari.types.LayerDataTuple:
    data = filters.sobel(layer.data)
    name = layer.name + '_sobel'
    contrast_limits = data.min(), data.max()
    dct = {
        **layer.as_layer_data_tuple()[1],
        'name': name,
        'contrast_limits': contrast_limits,
    }
    return (data, dct, 'image')
