import numpy as np
import pytest
from napari.layers import Image, Points
from napari.plugins._builtins import napari_write_image, napari_write_points
from napari.utils import io


@pytest.fixture(params=['image', 'points', 'points-with-properties'])
def layer_writer_and_data(request):
    if request.param == 'image':
        data = np.random.rand(20, 20)
        Layer = Image
        layer = Image(data)
        writer = napari_write_image
        extension = '.tif'

        def reader(path):
            return (
                io.imread(path),
                {},
            )

    elif request.param == 'points':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data)
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {},
            )

    elif request.param == 'points-with-properties':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data, properties={'values': np.random.rand(20)})
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {
                    'properties': {
                        io.read_csv(path)[1][3]: io.read_csv(path)[0][:, 3]
                    }
                },
            )

    else:
        return None, None, None, None, None

    layer_data = layer.as_layer_data_tuple()
    return writer, layer_data, extension, reader, Layer


@pytest.fixture
def layer_data_and_types():
    layers = [
        Image(np.random.rand(20, 20), name='ex_img'),
        Image(np.random.rand(20, 20)),
        Points(np.random.rand(20, 2), name='ex_pts'),
        Points(
            np.random.rand(20, 2), properties={'values': np.random.rand(20)}
        ),
    ]
    extensions = ['.tif', '.tif', '.csv', '.csv']
    layer_data = [l.as_layer_data_tuple() for l in layers]
    layer_types = [ld[2] for ld in layer_data]
    filenames = [l.name + e for l, e in zip(layers, extensions)]
    return layers, layer_data, layer_types, filenames
