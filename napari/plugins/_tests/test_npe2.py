from unittest.mock import Mock, patch
from npe2 import PluginManager, PluginManifest
from napari.plugins import _npe2
import yaml

import pytest

YAML = """
name: my-plugin
display_name: My Plugin
contributions:
  commands:
    - id: my-plugin.hello_world
      title: Hello World
    - id: my-plugin.another_command
      title: Another Command
    - id: my-plugin.some_reader
      title: Some Reader
    - id: my-plugin.url_reader
      title: URL Reader
    - id: my-plugin.my_writer
      title: My Multi-layer Writer
    - id: my-plugin.my_single_writer
      title: My single-layer Writer
    - id: my-plugin.generate_random_data
      title: Generate uniform random data
    - id: my-plugin.some_widget
      title: Create my widget
    - id: my-plugin.some_function_widget
      title: Create widget from my function
  configuration: # call it settings?
    properties:
      my_plugin.reader.lazy:
        type: boolean
        default: false
        title: Load lazily
        description: Whether to load images lazily with dask
  readers:
    - command: my-plugin.some_reader
      filename_patterns: ["*.fzy", "*.fzzy"]
      accepts_directories: true
    - command: my-plugin.url_reader
      filename_patterns: ["http://*", "https://*"]
      accepts_directories: false
  writers:
    - command: my-plugin.my_writer
      filename_extensions: ["*.tif", "*.tiff"]
      layer_types: ["image{2,4}", "tracks?"]
    - command: my-plugin.my_writer
      filename_extensions: ["*.pcd", "*.e57"]
      layer_types: ["points{1}", "surface+"]
    - command: my-plugin.my_single_writer
      filename_extensions: ["*.xyz"]
      layer_types: ["labels"]

  widgets:
    - command: my-plugin.some_widget
      display_name: My Widget
    - command: my-plugin.some_function_widget
      display_name: A Widget From a Function
      autogenerate: true
  menus:
    /napari/layer_context:
      - submenu: mysubmenu
      - command: my-plugin.hello_world
    mysubmenu:
      - command: my-plugin.another_command
      - command: my-plugin.affinder
  submenus:
    - id: mysubmenu
      label: My SubMenu
  themes:
    - label: "SampleTheme"
      id: "sample_theme"
      type: "dark"
      colors:
        canvas: "#000000"
        console: "#000000"
        background: "#272822"
        foreground: "#75715e"
        primary: "#cfcfc2"
        secondary: "#f8f8f2"
        highlight: "#e6db74"
        text: "#a1ef34"
        warning: "#f92672"
        current: "#66d9ef"
  sample_data:
    - display_name: Some Random Data (512 x 512)
      key: random_data
      command: my-plugin.generate_random_data
    - display_name: Random internet image
      key: internet_image
      uri: https://picsum.photos/1024
"""


@pytest.fixture
def sample_plugin():
    return PluginManifest(**yaml.safe_load(YAML))


@pytest.fixture
def mocked_npe2_pm(sample_plugin):
    with patch.object(PluginManager, 'discover'):
        _pm = PluginManager()
    _pm.register(sample_plugin)
    with patch('npe2.PluginManager.instance', return_value=_pm):
        yield _pm


def test_read(mocked_npe2_pm):
    with patch()
    assert _npe2.read(["some.fzzy"], stack=False) == (None, 'some plugin')



def test_write(layer_data_and_types):

    layers, layer_data, layer_types, filenames = layer_data_and_types
    with patch('napari.plugins._npe2.npe2.write') as mock:
        mock.side_effect = [[filenames[0]], []]
        result = _npe2.write_layers(filenames[0], [layers[0]])
        assert result == [filenames[0]]
        mock.assert_called_once()
        result = _npe2.write_layers("something.test", [layers[0]])
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
    result = _npe2.write_layers(filenames[0], [layers[0]], writer=writer)

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

        result = _npe2.get_widget_contribution('my-plugin')
        assert result[1] == 'My Widget'

        try:
            result = _npe2.get_widget_contribution(
                'my-plugin', widget_name="test plugin"
            )

        except KeyError:
            assert True

        result = _npe2.get_widget_contribution('my-plugin2')
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
        _npe2.populate_qmenu(menu, 'my-plugin')

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
        ext_str, writers = _npe2.file_extensions_string_for_layers(layers)

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

        readers = _npe2.get_readers("some.fzzy")
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

        sample_data = _npe2.get_sample_data('my-plugin', 'random_data')

        assert sample_data == (contrib.open, [])

        sample_data = _npe2.get_sample_data('my-plugin', 'other_data')
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
        for current_manifest in _npe2.iter_manifests():
            manifests.append(current_manifest)

        assert len(manifests) == len(input_manifests)
