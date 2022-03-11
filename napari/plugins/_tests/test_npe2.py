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
)


def test_read():

    with patch('napari.plugins._npe2.read_get_reader') as mock1, patch(
        'napari.plugins._npe2._FakeHookimpl'
    ) as mock2:
        mock_reader = Mock()
        mock_reader.plugin_name = 'some plugin'
        mock1.return_value = (None, mock_reader())
        mock2.return_value = 'some plugin'
        assert read(["some.fzzy"], stack=False) == (None, 'some plugin')
        mock1.assert_called_once()
        mock2.assert_called_once()


def test_write(layer_data_and_types):

    layers, layer_data, layer_types, filenames = layer_data_and_types
    with patch('napari.plugins._npe2.npe2.write') as mock:
        mock.side_effect = [[filenames[0]], []]
        result = write_layers(filenames[0], [layers[0]])
        assert result == [filenames[0]]
        mock.assert_called_once()
        result = write_layers("something.test", [layers[0]])
        assert result == []
        assert mock.call_count == 2

    writer = Mock()
    ltc1 = Mock()
    ltc1.max = Mock(return_value=1)
    ltc2 = Mock()
    ltc2.max = Mock(return_value=0)
    writer.layer_type_constraints = Mock()
    writer.layer_type_constraints.return_value = [
        ltc1,
        ltc2,
        ltc2,
        ltc2,
        ltc2,
        ltc2,
        ltc2,
    ]

    # is the following cheating?
    writer.exec = Mock(return_value=[filenames[0]])
    result = write_layers(filenames[0], [layers[0]], writer=writer)

    assert result == [filenames[0]]


def test_get_widget_contribution():

    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:

        contrib = Mock()
        contrib.plugin_name = 'my-plugin'
        contrib.display_name = 'My Widget'
        contrib.get_callable = Mock(return_value=None)
        # contrib.get_callable.
        instance = Mock()
        instance.iter_widgets = Mock(return_value=[contrib])
        # instance.iter_widgets.
        mock.return_value = instance

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


def test_populate_qmenu():
    '''Tests for populate_qmenu.'''

    # Tests the whole method. the first run through, it will add a menu,
    # but will then call populate_qmenu will call itself again to add the submenu
    # to that menu added.
    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:
        instance = Mock()

        class Item1:
            # Using this class to return an object for both the
            # item and subm_contrib in the code.
            command = 'run plugin'
            submenu = 'My submenu'
            id = 'my plugin'
            label = 'my label'

        class Item2:
            command = 'run plugin'

        item1 = Item1()
        item2 = Item2()
        side_effect = [[item1], [item2]]
        instance.iter_menu = Mock(side_effect=side_effect)
        instance.get_submenu = Mock(return_value=Item1())
        menu = Mock()
        submenu = Mock()
        submenu.addAction = Mock()
        menu.addMenu = Mock(return_value=submenu)

        mock.return_value = instance
        populate_qmenu(menu, 'my-plugin')

        submenu.addAction.assert_called_once()
        assert instance.iter_menu.call_count == len(side_effect)
        menu.addMenu.assert_called_once()


def test_file_extensions_string_for_layers(layer_data_and_types):

    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:
        instance = Mock()
        writer1 = Mock()
        writer1.display_name = 'image writer'
        writer1.command = None
        writer1.filename_extensions = ['.jpg', '.giff']
        writer2 = Mock()
        writer2.display_name = 'text writer'
        writer2.command = None
        writer2.filename_extensions = ['.txt']
        instance.iter_compatible_writers = Mock()
        instance.iter_compatible_writers.return_value = [writer1, writer2]
        manifest = Mock()
        manifest.display_name = 'my plugin'
        instance.get_manifest = Mock(return_value=manifest)
        mock.return_value = instance

        layers, layer_data, layer_types, filenames = layer_data_and_types
        ext_str, writers = file_extensions_string_for_layers(layers)

        assert len(writers) == 2
        assert (
            ext_str
            == 'my plugin image writer (*.jpg *.giff);;my plugin text writer (*.txt)'
        )


def test_get_readers():

    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:

        reader = Mock()
        reader.plugin_name = 'my-plugin'
        reader.command = None
        instance = Mock()
        instance.iter_compatible_readers = Mock()
        instance.iter_compatible_readers.return_value = [reader]
        manifest = Mock()
        manifest.display_name = 'My Plugin'
        instance.get_manifest = Mock(return_value=manifest)
        mock.return_value = instance

        readers = get_readers("some.fzzy")
        assert readers['My Plugin'] == 'my-plugin'


def test_get_sample_data(layer_data_and_types):
    import numpy as np

    layers, layer_data, layer_types, filenames = layer_data_and_types
    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:

        instance = Mock()
        contrib = Mock()
        contrib.key = 'random_data'
        contrib.open = Mock(return_value=(np.random.rand(10, 10), 'reader'))

        instance.iter_sample_data = Mock(
            return_value=[('my-plugin', [contrib])]
        )

        mock.return_value = instance

        sample_data = get_sample_data('my-plugin', 'random_data')

        assert sample_data == (contrib.open, [])

        sample_data = get_sample_data('my-plugin', 'other_data')
        avail = instance.iter_sample_data()
        plugin_name = avail[0][0]
        key = avail[0][1][0].key
        assert sample_data == (None, [(plugin_name, key)])

    with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:
        # instance1 is to test for newer npe versions that have iter_manifests method
        instance1 = Mock()
        manifest = Mock()

        input_manifests = [manifest, manifest, manifest]
        instance1.iter_manifests = Mock(return_value=input_manifests)

        # instance2 is to test for older npe versions that have pm._manifests
        instance2 = Mock()
        instance2._manifests = {'m1': manifest, 'm2': manifest}

        # need to have appropriate instances for mock at each call.
        side_effects = []
        for val in input_manifests:
            side_effects.append(instance1)

        for val in instance2._manifests.values():
            side_effects.append(instance2)

        mock.side_effect = side_effects

        # test newer npe2 version
        manifests = []
        for current_manifest in iter_manifests():
            manifests.append(current_manifest)

        assert len(manifests) == len(input_manifests)
