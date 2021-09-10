from pathlib import Path

import numpy as np
import pytest
from napari_plugin_engine import napari_hook_implementation

import napari
from napari.layers._source import Source
from napari.viewer import ViewerModel


def test_sample_hook(napari_plugin_manager):

    viewer = ViewerModel()
    napari_plugin_manager.discover_sample_data()

    with pytest.raises(KeyError) as e:
        viewer.open_sample('test_plugin', 'random data')

        print([str(e)])
    assert (
        "Plugin 'test_plugin' does not provide sample data named 'random data'"
        in str(e)
    )

    def _generate_random_data(shape=(512, 512)):
        data = np.random.rand(*shape)
        return [(data, {'name': 'random data'})]

    LOGO = Path(napari.__file__).parent / 'resources' / 'logo.png'

    class test_plugin:
        @napari_hook_implementation
        def napari_provide_sample_data():
            return {
                'random data': _generate_random_data,
                'napari logo': LOGO,
                'samp_key': {
                    'data': _generate_random_data,
                    'display_name': 'I look gorgeous in the menu!',
                },
            }

    napari_plugin_manager.register(test_plugin)

    reg = napari_plugin_manager._sample_data['test_plugin']
    assert reg['random data']['data'] == _generate_random_data
    assert reg['random data']['display_name'] == 'random data'
    assert reg['napari logo']['data'] == LOGO
    assert reg['napari logo']['display_name'] == 'napari logo'
    assert reg['samp_key']['data'] == _generate_random_data
    assert reg['samp_key']['display_name'] == 'I look gorgeous in the menu!'

    assert len(viewer.layers) == 0
    viewer.open_sample('test_plugin', 'random data')
    assert viewer.layers[-1].source == Source(
        path=None, reader_plugin=None, sample=('test_plugin', 'random data')
    )
    assert len(viewer.layers) == 1
    viewer.open_sample('test_plugin', 'napari logo')
    assert viewer.layers[-1].source == Source(
        path=str(LOGO),
        reader_plugin='builtins',
        sample=('test_plugin', 'napari logo'),
    )
    assert len(viewer.layers) == 2
    viewer.open_sample('test_plugin', 'samp_key')
    assert viewer.layers[-1].source == Source(
        sample=('test_plugin', 'samp_key')
    )
    assert len(viewer.layers) == 3

    # test calling with kwargs
    viewer.open_sample('test_plugin', 'samp_key', shape=(256, 256))
    assert len(viewer.layers) == 4
    assert viewer.layers[-1].source == Source(
        sample=('test_plugin', 'samp_key')
    )
