from types import MethodType
from unittest.mock import MagicMock, patch

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
    - id: {0}.some_reader
      title: Some Reader
    - id: {0}.my_writer
      title: Image Writer
    - id: {0}.generate_random_data
      title: Generate uniform random data
    - id: {0}.some_widget
      title: Create my widget
  readers:
    - command: {0}.some_reader
      filename_patterns: ["*.fzy", "*.fzzy"]
      accepts_directories: true
  writers:
    - command: {0}.my_writer
      filename_extensions: ["*.tif", "*.tiff"]
      layer_types: ["image"]
  widgets:
    - command: {0}.some_widget
      display_name: My Widget
  menus:
    /napari/layer_context:
      - submenu: mysubmenu
      - command: {0}.hello_world
    mysubmenu:
      - command: {0}.hello_world
  submenus:
    - id: mysubmenu
      label: My SubMenu
  themes:
    - label: "SampleTheme"
      id: "sample_theme"
      type: "dark"
      colors:
        background: "#272822"
        foreground: "#75715e"
  sample_data:
    - display_name: Some Random Data (512 x 512)
      key: random_data
      command: {0}.generate_random_data
    - display_name: Random internet image
      key: internet_image
      uri: https://picsum.photos/1024
"""


@pytest.fixture
def sample_plugin():
    return PluginManifest(**yaml.safe_load(YAML.format(PLUGIN_NAME)))


@pytest.fixture
def mock_pm(sample_plugin):
    mock_reg = MagicMock()
    with patch.object(PluginManager, 'discover'):
        _pm = PluginManager(reg=mock_reg)
    _pm.register(sample_plugin)
    with patch('npe2.PluginManager.instance', return_value=_pm):
        yield _pm


def test_read(mock_pm):
    _, hookimpl = _npe2.read(["some.fzzy"], stack=False)
    mock_pm.commands.get.assert_called_once_with(f'{PLUGIN_NAME}.some_reader')
    assert hookimpl.plugin_name == PLUGIN_NAME

    mock_pm.commands.get.reset_mock()
    _, hookimpl = _npe2.read(["some.fzzy"], stack=True)
    mock_pm.commands.get.assert_called_once_with(f'{PLUGIN_NAME}.some_reader')

    mock_pm.commands.get.reset_mock()
    assert _npe2.read(["some.randomext"], stack=True) is None
    mock_pm.commands.get.assert_not_called()


def test_write(mock_pm: PluginManager):
    # saving an image without a writer goes straight to npe2.write
    # it will use our plugin writer
    image = Image(np.random.rand(20, 20), name='ex_img')
    _npe2.write_layers('some_file.tif', [image])
    mock_pm.commands.get.assert_called_once_with(f'{PLUGIN_NAME}.my_writer')

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
    writer.exec.return_value = ['']
    assert _npe2.write_layers('some_file.tif', [points], writer=writer) == ['']
    mock_pm.commands.get.assert_not_called()
    writer.exec.assert_called_once()
    assert writer.exec.call_args_list[0].kwargs['args'][0] == 'some_file.tif'


def test_get_widget_contribution(mock_pm):
    # calling with plugin alone
    (_, display_name) = _npe2.get_widget_contribution(PLUGIN_NAME)
    mock_pm.commands.get.assert_called_once_with('my-plugin.some_widget')
    assert display_name == 'My Widget'

    # calling with plugin but wrong widget name provides a useful error msg
    with pytest.raises(KeyError) as e:
        _npe2.get_widget_contribution(PLUGIN_NAME, 'Not a widget')
    assert (
        f"Plugin {PLUGIN_NAME!r} does not provide a widget named 'Not a widget'"
        in str(e.value)
    )

    # calling with a non-existent plugin just returns None
    mock_pm.commands.get.reset_mock()
    assert not _npe2.get_widget_contribution('not-a-thing')
    mock_pm.commands.get.assert_not_called()


def test_populate_qmenu(mock_pm):
    menu = MagicMock()
    _npe2.populate_qmenu(menu, '/napari/layer_context')
    assert menu.addMenu.called_once_with('My SubMenu')
    assert menu.addAction.called_once_with('Hello World')


def test_file_extensions_string_for_layers(mock_pm):
    layers = [Image(np.random.rand(20, 20), name='ex_img')]
    label, writers = _npe2.file_extensions_string_for_layers(layers)
    assert label == 'My Plugin (*.tif *.tiff)'
    writer = mock_pm.get_manifest(PLUGIN_NAME).contributions.writers[0]
    assert writers == [writer]


def test_get_readers(mock_pm):
    assert _npe2.get_readers("some.fzzy") == {PLUGIN_NAME: 'My Plugin'}


def test_iter_manifest(mock_pm):
    for i in _npe2.iter_manifests():
        assert isinstance(i, PluginManifest)


def test_get_sample_data(mock_pm):
    samples = mock_pm.get_manifest(PLUGIN_NAME).contributions.sample_data

    opener, _ = _npe2.get_sample_data(PLUGIN_NAME, 'random_data')
    assert isinstance(opener, MethodType) and opener.__self__ is samples[0]

    opener, _ = _npe2.get_sample_data(PLUGIN_NAME, 'internet_image')
    assert isinstance(opener, MethodType) and opener.__self__ is samples[1]

    opener, avail = _npe2.get_sample_data('not-a-plugin', 'nor-a-sample')
    assert opener is None
    assert avail == [
        (PLUGIN_NAME, 'random_data'),
        (PLUGIN_NAME, 'internet_image'),
    ]


def test_sample_iterator(mock_pm):
    samples = list(_npe2.sample_iterator())
    assert samples
    for plugin, contribs in samples:
        assert isinstance(plugin, str)
        assert isinstance(contribs, dict)
        assert contribs
        for i in contribs.values():
            assert 'data' in i
            assert 'display_name' in i


def test_widget_iterator(mock_pm):
    wdgs = list(_npe2.widget_iterator())
    assert wdgs == [('dock', ('my-plugin', ['My Widget']))]
