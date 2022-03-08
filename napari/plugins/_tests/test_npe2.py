from os import TMP_MAX
from re import A
from npe2 import PluginManager
from npe2.types import FullLayerData
from unittest.mock import Mock, patch

from ...plugins._npe2 import (
    file_extensions_string_for_layers,
    get_readers,
    get_sample_data,
    get_widget_contribution,
    iter_manifests,
    populate_qmenu,
    read,
    write_layers,
    read_get_reader,
    _FakeHookimpl,
    npe2
    
)

# from PyQt5.QtWidgets import QMenu


    

def test_read():

    with patch('napari.plugins._npe2.read_get_reader') as mock1, patch(
        'napari.plugins._npe2._FakeHookimpl') as mock2:
        mock_reader = Mock()
        mock_reader.plugin_name = 'some plugin'
        mock1.return_value = (None, mock_reader())
        mock2.return_value = 'some plugin'
        assert read(["some.fzzy"], stack=False) == (None, 'some plugin')
        mock1.assert_called_once()
        mock2.assert_called_once()

def test_write(layer_data_and_types):
    
    layers, layer_data, layer_types, filenames = layer_data_and_types
    with patch('napari.plugins._npe2.npe2.write') as mock1:
        mock1.side_effect = [[filenames[0]], []]
        result = write_layers(filenames[0], [layers[0]])
        assert result == [filenames[0]]
        mock1.assert_called_once()
        result = write_layers("something.test", [layers[0]])
        assert result == []
        assert mock1.call_count == 2
    
    writer = Mock()
    ltc1 = Mock()
    ltc1.max = Mock()
    ltc1.max.return_value = 1
    ltc = Mock()
    ltc.max = Mock()
    ltc.max.return_value = 0
    writer.layer_type_constraints = Mock()
    writer.layer_type_constraints.return_value = [ltc1, ltc, ltc, ltc, ltc, ltc, ltc]
    
    writer.exec = Mock()
    # is the following cheating?
    writer.exec.return_value = [filenames[0]]
    result = write_layers(filenames[0], [layers[0]], writer=writer)
        
    result = write_layers(filenames[0], [layers[0]], writer=writer)
    
    assert result == [filenames[0]]


def test_get_widget_contribution():

    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock1:

        contrib = Mock()
        contrib.plugin_name = 'my-plugin'
        contrib.display_name = 'My Widget'
        contrib.get_callable = Mock()
        contrib.get_callable.return_value = None
        instance = Mock()
        instance.iter_widgets = Mock()
        instance.iter_widgets.return_value = [contrib]
        mock1.return_value = instance
        
        result = get_widget_contribution('my-plugin')
        assert result[1] == 'My Widget'

        try:
            result = get_widget_contribution(
                'my-plugin', widget_name="test plugin"
            )

        except KeyError:
            assert True

        result = get_widget_contribution('my-plugin2')
        assert result is None


def test_populate_qmenu(make_napari_viewer):

    
    # not sure if this is going anywhere promising...

    # with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock1:
    #     cmd = Mock()
    #     cmd.title = 'My Plugin'
    #     cmd.exec = Mock()
    #     # cmd.exec.return_value = None
    #     mock1.get_command = Mock()
    #     mock1.get_command.return_value = cmd

    #     menu = Mock()
    #     menu.addAction = Mock()
    #     menu.actions = Mock()
    #     menu.actions.return_value = []
    #     menu.addAction.return_value = menu.actions().append(action)

        viewer = make_napari_viewer()
        menu = viewer.window.plugins_menu
        actions1 = menu.actions()
        # these actions are already here.
        populate_qmenu(menu, 'my-plugin')
        actions2 = menu.actions()
        assert len(actions1) == len(actions2)

    
    # menu2 = QMenu()
    # populate_qmenu(menu2, '/napari/layer_context')
    # texts = [a.text() for a in menu2.actions()]

    # assert texts[0] == 'My SubMenu'


def test_file_extensions_string_for_layers(layer_data_and_types):

    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock1:
        instance = Mock()
        writer = Mock()
        writer.display_name = 'image writer'
        writer.command = None
        writer.filename_extensions = ['.jpg', '.giff']
        writer2 = Mock()
        writer2.display_name = 'text writer'
        writer2.command = None
        writer2.filename_extensions = ['.txt']
        instance.iter_compatible_writers = Mock()
        instance.iter_compatible_writers.return_value = [writer, writer2]
        instance.get_manifest = Mock()
        manifest = Mock()
        manifest.display_name = 'my plugin'
        instance.get_manifest.return_value = manifest
        mock1.return_value = instance
        
        layers, layer_data, layer_types, filenames = layer_data_and_types
        ext_str, writers = file_extensions_string_for_layers(layers)
        
        assert len(writers) == 2
        assert ext_str == 'my plugin image writer (*.jpg *.giff);;my plugin text writer (*.txt)'



def test_get_readers():
    
    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock1:

        reader = Mock()
        reader.plugin_name = 'my-plugin'
        reader.command = None
        instance = Mock()
        instance.iter_compatible_readers = Mock()
        instance.iter_compatible_readers.return_value = [reader]
        instance.get_manifest = Mock()
        manifest = Mock()
        manifest.display_name = 'My Plugin'
        instance.get_manifest.return_value = manifest
        mock1.return_value = instance

        
        readers = get_readers("some.fzzy")
        assert readers['My Plugin'] == 'my-plugin'

def test_get_sample_data(layer_data_and_types):
    import numpy as np
    layers, layer_data, layer_types, filenames = layer_data_and_types
    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock1:
        
        c = Mock()
        c.key = 'random_data'
        c.open = Mock()
        c.open.return_value = (np.random.rand(10,10), 'reader') # (layer_data, reader)
        instance = Mock()
        instance._contrib._samples.get = Mock()
        instance._contrib._samples.get.return_value = [c]
        mock1.return_value = instance

        # test correct sample name
        sample_data = get_sample_data('my-plugin', 'random_data')
        output = sample_data[0]()
        assert output[0].shape == (10, 10)

        # test incorrect sample name
        instance.iter_sample_data = Mock()
        instance.iter_sample_data.return_value = []
        sample_data = get_sample_data('my-plugin', 'random_data2')
        assert sample_data[0] is None


def test_iter_manifests():
    
    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock1:
        instance = Mock()
        manifest = Mock()
        manifest.name = 'my-plugin'
        instance._manifests = {'m1': manifest, 'm2': manifest}
        mock1.return_value = instance

        manifests = []
        for manifest in iter_manifests():
            manifests.append(manifest)

        assert len(manifests) == 2
        assert manifests[0].name == 'my-plugin' 