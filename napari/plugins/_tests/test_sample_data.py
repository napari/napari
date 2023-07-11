from pathlib import Path

import numpy as np
import pytest
from npe2 import DynamicPlugin
from npe2.manifest.contributions import SampleDataURI

import napari
from napari.layers._source import Source
from napari.viewer import ViewerModel

LOGO = str(Path(napari.__file__).parent / 'resources' / 'logo.png')


def test_sample_hook(builtins, tmp_plugin: DynamicPlugin):
    viewer = ViewerModel()
    NAME = tmp_plugin.name
    KEY = 'random data'
    with pytest.raises(KeyError, match=f"Plugin {NAME!r} does not provide"):
        viewer.open_sample(NAME, KEY)

    @tmp_plugin.contribute.sample_data(key=KEY)
    def _generate_random_data(shape=(512, 512)):
        data = np.random.rand(*shape)
        return [(data, {'name': KEY})]

    tmp_plugin.manifest.contributions.sample_data.append(
        SampleDataURI(uri=LOGO, key='napari logo', display_name='Napari logo')
    )

    assert len(viewer.layers) == 0
    viewer.open_sample(NAME, KEY)
    assert viewer.layers[-1].source == Source(
        path=None, reader_plugin=None, sample=(NAME, KEY)
    )
    assert len(viewer.layers) == 1
    viewer.open_sample(NAME, 'napari logo')
    assert viewer.layers[-1].source == Source(
        path=LOGO, reader_plugin='napari', sample=(NAME, 'napari logo')
    )

    # test calling with kwargs
    viewer.open_sample(NAME, KEY, shape=(256, 256))
    assert len(viewer.layers) == 3
    assert viewer.layers[-1].source == Source(sample=(NAME, KEY))


def test_sample_uses_reader_plugin(builtins, tmp_plugin, tmp_path):
    viewer = ViewerModel()
    NAME = tmp_plugin.name
    tmp_plugin.manifest.contributions.sample_data = [
        SampleDataURI(
            uri=LOGO,
            key='napari logo',
            display_name='Napari logo',
            reader_plugin='gibberish',
        )
    ]
    # if we don't pass a plugin, the declared reader_plugin is tried
    with pytest.raises(ValueError) as e:
        viewer.open_sample(NAME, 'napari logo')
    assert "There is no registered plugin named 'gibberish'" in str(e)

    # if we pass a plugin, it overrides the declared one
    viewer.open_sample(NAME, 'napari logo', reader_plugin='napari')
    assert len(viewer.layers) == 1

    # if we pass a plugin that fails, we get the right error message
    fake_uri = tmp_path / 'fakepath.png'
    fake_uri.touch()
    tmp_plugin.manifest.contributions.sample_data = [
        SampleDataURI(
            uri=str(fake_uri),
            key='fake sample',
            display_name='fake sample',
            reader_plugin='gibberish',
        )
    ]
    with pytest.raises(ValueError) as e:
        viewer.open_sample(NAME, 'fake sample', reader_plugin='napari')
    assert (
        f"Chosen reader napari failed to open sample. Plugin {NAME} declares gibberish"
        in str(e)
    )
