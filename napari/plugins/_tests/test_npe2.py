from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import yaml
from npe2 import PluginManager, PluginManifest

from napari.layers import Image, Points
from napari.plugins import _npe2

PLUGIN_NAME = 'my-plugin'

YAML = """
name: {0}
display_name: My Plugin
contributions:
  commands:
    - id: {0}.hello_world
      title: Hello World
    - id: {0}.another_command
      title: Another Command
    - id: {0}.some_reader
      title: Some Reader
    - id: {0}.url_reader
      title: URL Reader
    - id: {0}.my_writer
      title: My Multi-layer Writer
    - id: {0}.my_single_writer
      title: My single-layer Writer
    - id: {0}.generate_random_data
      title: Generate uniform random data
    - id: {0}.some_widget
      title: Create my widget
    - id: {0}.some_function_widget
      title: Create widget from my function
  configuration: # call it settings?
    properties:
      my_plugin.reader.lazy:
        type: boolean
        default: false
        title: Load lazily
        description: Whether to load images lazily with dask
  readers:
    - command: {0}.some_reader
      filename_patterns: ["*.fzy", "*.fzzy"]
      accepts_directories: true
    - command: {0}.url_reader
      filename_patterns: ["http://*", "https://*"]
      accepts_directories: false
  writers:
    - command: {0}.my_writer
      filename_extensions: ["*.tif", "*.tiff"]
      layer_types: ["image"]
    - command: {0}.my_writer
      filename_extensions: ["*.pcd", "*.e57"]
      layer_types: ["points{{1}}", "surface+"]
    - command: {0}.my_single_writer
      filename_extensions: ["*.xyz"]
      layer_types: ["labels"]

  widgets:
    - command: {0}.some_widget
      display_name: My Widget
    - command: {0}.some_function_widget
      display_name: A Widget From a Function
      autogenerate: true
  menus:
    /napari/layer_context:
      - submenu: mysubmenu
      - command: {0}.hello_world
    mysubmenu:
      - command: {0}.another_command
      - command: {0}.affinder
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
      command: {0}.generate_random_data
    - display_name: Random internet image
      key: internet_image
      uri: https://picsum.photos/1024
""".format(
    PLUGIN_NAME
)


@pytest.fixture
def sample_plugin():
    return PluginManifest(**yaml.safe_load(YAML))


@pytest.fixture
def mock_pm(sample_plugin):
    mock_reg = MagicMock()
    with patch.object(PluginManager, 'discover'):
        _pm = PluginManager(reg=mock_reg)
    _pm.register(sample_plugin)
    with patch('npe2.PluginManager.instance', return_value=_pm):
        yield _pm


def test_read(mock_pm):
    data, hookimpl = _npe2.read(["some.fzzy"], stack=False)
    mock_pm.commands.get.assert_called_once_with('my-plugin.some_reader')
    assert hookimpl.plugin_name == PLUGIN_NAME


def test_write(mock_pm: PluginManager):
    # saving an image without a writer goes straight to npe2.write
    # it will use our plugin writer
    image = Image(np.random.rand(20, 20), name='ex_img')
    _npe2.write_layers('some_file.tif', [image])
    mock_pm.commands.get.assert_called_once_with('my-plugin.my_writer')

    # points won't trigger our sample writer
    mock_pm.commands.get.reset_mock()
    points = Points(np.random.rand(20, 2), name='ex_points')
    _npe2.write_layers('some_file.tif', [points])
    mock_pm.commands.get.assert_not_called()

    # calling _npe2.write_layers with a specific writer contribution should
    # directly execute the writer.exec with arguments appropriate for the
    # writer spec (single or multi-writer)
    mock_pm.commands.get.reset_mock()
    writer = mock_pm.get_manifest(PLUGIN_NAME).contributions.writers[0]
    writer = MagicMock(wraps=writer)
    writer.exec.return_value = ''
    assert _npe2.write_layers('some_file.tif', [points], writer=writer) == ['']
    mock_pm.commands.get.assert_not_called()
    expected_args = ('some_file.tif', *points.as_layer_data_tuple()[:2])
    writer.exec.assert_called_once_with(args=expected_args)


def test_get_widget_contribution(mock_pm):
    # calling with plugin alone
    (_, display_name) = _npe2.get_widget_contribution('my-plugin')
    mock_pm.commands.get.assert_called_once_with('my-plugin.some_widget')
    assert display_name == 'My Widget'

    # calling with plugin but wrong widget name provides a useful error msg
    with pytest.raises(KeyError) as e:
        _npe2.get_widget_contribution('my-plugin', 'Not a widget')
    assert (
        "Plugin 'my-plugin' does not provide a widget named 'Not a widget'"
        in str(e.value)
    )

    # calling with a non-existent plugin just returns None
    mock_pm.commands.get.reset_mock()
    assert not _npe2.get_widget_contribution('not-a-thing')
    mock_pm.commands.get.assert_not_called()


def test_populate_qmenu(mock_pm):
    menu = MagicMock()
    _npe2.populate_qmenu(menu, '/napari/layer_context')
    assert menu.addAction.called_once_with('Hello World')

    menu.reset_mock()
    _npe2.populate_qmenu(menu, 'mysubmenu')
    breakpoint()
    assert menu.addMenu.called_once_with('Hello World')


#     '''Tests for populate_qmenu.'''

#     # Tests the whole method. the first run through, it will add a menu,
#     # but will then call populate_qmenu will call itself again to add the submenu
#     # to that menu added.
#     with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:
#         instance = Mock()

#         class Item1:
#             # Using this class to return an object for both the
#             # item and subm_contrib in the code.
#             command = 'run plugin'
#             submenu = 'My submenu'
#             id = 'my plugin'
#             label = 'my label'

#         class Item2:
#             command = 'run plugin'

#         item1 = Item1()
#         item2 = Item2()
#         side_effect = [[item1], [item2]]
#         instance.iter_menu = Mock(side_effect=side_effect)
#         instance.get_submenu = Mock(return_value=Item1())
#         menu = Mock()
#         submenu = Mock()
#         submenu.addAction = Mock()
#         menu.addMenu = Mock(return_value=submenu)

#         mock.return_value = instance
#         _npe2.populate_qmenu(menu, 'my-plugin')

#         submenu.addAction.assert_called_once()
#         assert instance.iter_menu.call_count == len(side_effect)
#         menu.addMenu.assert_called_once()


# def test_file_extensions_string_for_layers(layer_data_and_types):

#     with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:
#         instance = Mock()
#         writer1 = Mock()
#         writer1.display_name = 'image writer'
#         writer1.command = None
#         writer1.filename_extensions = ['.jpg', '.giff']
#         writer2 = Mock()
#         writer2.display_name = 'text writer'
#         writer2.command = None
#         writer2.filename_extensions = ['.txt']
#         instance.iter_compatible_writers = Mock()
#         instance.iter_compatible_writers.return_value = [writer1, writer2]
#         manifest = Mock()
#         manifest.display_name = 'my plugin'
#         instance.get_manifest = Mock(return_value=manifest)
#         mock.return_value = instance

#         layers, layer_data, layer_types, filenames = layer_data_and_types
#         ext_str, writers = _npe2.file_extensions_string_for_layers(layers)

#         assert len(writers) == 2
#         assert (
#             ext_str
#             == 'my plugin image writer (*.jpg *.giff);;my plugin text writer (*.txt)'
#         )


# def test_get_readers():

#     with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:

#         reader = Mock()
#         reader.plugin_name = 'my-plugin'
#         reader.command = None
#         instance = Mock()
#         instance.iter_compatible_readers = Mock()
#         instance.iter_compatible_readers.return_value = [reader]
#         manifest = Mock()
#         manifest.display_name = 'My Plugin'
#         instance.get_manifest = Mock(return_value=manifest)
#         mock.return_value = instance

#         readers = _npe2.get_readers("some.fzzy")
#         assert readers['My Plugin'] == 'my-plugin'


# def test_get_sample_data(layer_data_and_types):
#     import numpy as np

#     layers, layer_data, layer_types, filenames = layer_data_and_types
#     with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:

#         instance = Mock()
#         contrib = Mock()
#         contrib.key = 'random_data'
#         contrib.open = Mock(return_value=(np.random.rand(10, 10), 'reader'))

#         instance.iter_sample_data = Mock(
#             return_value=[('my-plugin', [contrib])]
#         )

#         mock.return_value = instance

#         sample_data = _npe2.get_sample_data('my-plugin', 'random_data')

#         assert sample_data == (contrib.open, [])

#         sample_data = _npe2.get_sample_data('my-plugin', 'other_data')
#         avail = instance.iter_sample_data()
#         plugin_name = avail[0][0]
#         key = avail[0][1][0].key
#         assert sample_data == (None, [(plugin_name, key)])

#     with patch('napari.plugins._npe2.npe2.PluginManager.instance') as mock:
#         # instance1 is to test for newer npe versions that have iter_manifests method
#         instance1 = Mock()
#         manifest = Mock()

#         input_manifests = [manifest, manifest, manifest]
#         instance1.iter_manifests = Mock(return_value=input_manifests)

#         # instance2 is to test for older npe versions that have pm._manifests
#         instance2 = Mock()
#         instance2._manifests = {'m1': manifest, 'm2': manifest}

#         # need to have appropriate instances for mock at each call.
#         side_effects = []
#         for val in input_manifests:
#             side_effects.append(instance1)

#         for val in instance2._manifests.values():
#             side_effects.append(instance2)

#         mock.side_effect = side_effects

#         # test newer npe2 version
#         manifests = []
#         for current_manifest in _npe2.iter_manifests():
#             manifests.append(current_manifest)

#         assert len(manifests) == len(input_manifests)
