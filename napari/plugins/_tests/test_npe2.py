from npe2 import PluginManager
from npe2.types import FullLayerData
from PyQt5.QtWidgets import QMenu

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

null_image: FullLayerData = ([], {}, "image")


def test_read(uses_sample_plugin):
    assert read("some.fzzy")[0] == [(None,)]

    try:
        read("some.test")
    except ValueError:
        assert True


def test_write(uses_sample_plugin, layer_data_and_types):

    layers, layer_data, layer_types, filenames = layer_data_and_types
    result = write_layers(filenames[0], [layers[0]])
    assert result == [filenames[0]]

    result = write_layers("something.test", [layers[0]])
    assert result == []

    _pm = PluginManager.instance()
    writer, new_path = _pm.get_writer(
        filenames[0], layer_types=[layer_types[0]]
    )
    result = write_layers(filenames[0], [layers[0]], writer=writer)
    assert result == [filenames[0]]


def test_get_widget_contribution(uses_sample_plugin):
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


def test_populate_qmenu(uses_sample_plugin, make_napari_viewer):

    viewer = make_napari_viewer()
    menu = viewer.window.plugins_menu
    actions1 = menu.actions()
    # these actions are already here.
    populate_qmenu(menu, 'my-plugin')
    actions2 = menu.actions()
    assert len(actions1) == len(actions2)

    menu2 = QMenu()
    populate_qmenu(menu2, '/napari/layer_context')
    texts = [a.text() for a in menu2.actions()]

    assert texts[0] == 'My SubMenu'


def test_file_extensions_string_for_layers(
    uses_sample_plugin, layer_data_and_types
):

    layers, layer_data, layer_types, filenames = layer_data_and_types
    ext_str, writers = file_extensions_string_for_layers(layers)

    print(ext_str)
    print(writers)

    assert len(writers) == 1


def test_get_readers(uses_sample_plugin):
    readers = get_readers("some.fzzy")
    assert readers['My Plugin'] == 'my-plugin'


def test_get_sample_data(uses_sample_plugin):

    # test correct sample name
    sample_data = get_sample_data('my-plugin', 'random_data')

    tmp = sample_data[0]()

    assert tmp[0].shape == (10, 10)

    # test incorrect sample name
    sample_data = get_sample_data('my-plugin', 'random_data2')
    assert sample_data[0] is None


def test_iter_manifests(uses_sample_plugin):
    for manifest in iter_manifests():
        if manifest.name == 'my-plugin':
            break
    assert manifest.name == 'my-plugin'
