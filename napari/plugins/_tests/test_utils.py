from npe2 import DynamicPlugin

from napari.plugins.utils import (
    get_all_readers,
    get_filename_patterns_for_reader,
    get_potential_readers,
    get_preferred_reader,
)
from napari.settings import get_settings


def test_get_preferred_reader_no_readers():
    get_settings().plugins.extension2reader = {}
    reader = get_preferred_reader('my_file.tif')
    assert reader is None


def test_get_preferred_reader_for_extension():
    get_settings().plugins.extension2reader = {'*.tif': 'fake-plugin'}
    reader = get_preferred_reader('my_file.tif')
    assert reader == 'fake-plugin'


def test_get_preferred_reader_complex_pattern():
    get_settings().plugins.extension2reader = {
        'my-specific-folder/*.tif': 'fake-plugin'
    }
    reader = get_preferred_reader('my-specific-folder/my_file.tif')
    assert reader == 'fake-plugin'


def test_get_preferred_reader_no_extension():
    assert get_preferred_reader('my_file') is None


def test_get_potential_readers_gives_napari(
    builtins, tmp_plugin: DynamicPlugin
):
    @tmp_plugin.contribute.reader(filename_patterns=['*.tif'])
    def read_tif(path):
        ...

    readers = get_potential_readers('my_file.tif')
    assert 'napari' in readers
    assert 'builtins' not in readers


def test_get_potential_readers_finds_readers(tmp_plugin: DynamicPlugin):
    tmp2 = tmp_plugin.spawn(register=True)

    @tmp_plugin.contribute.reader(filename_patterns=['*.tif'])
    def read_tif(path):
        ...

    @tmp2.contribute.reader(filename_patterns=['*.*'])
    def read_all(path):
        ...

    readers = get_potential_readers('my_file.tif')
    assert len(readers) == 2


def test_get_potential_readers_none_available():
    assert not get_potential_readers('my_file.fake')


def test_get_potential_readers_plugin_name_disp_name(
    tmp_plugin: DynamicPlugin,
):
    @tmp_plugin.contribute.reader(filename_patterns=['*.fake'])
    def read_tif(path):
        ...

    readers = get_potential_readers('my_file.fake')
    assert readers[tmp_plugin.name] == tmp_plugin.display_name


def test_get_all_readers_gives_napari(builtins):
    npe2_readers, npe1_readers = get_all_readers()
    assert len(npe1_readers) == 0
    assert len(npe2_readers) == 1
    assert 'napari' in npe2_readers


def test_get_all_readers(tmp_plugin: DynamicPlugin):
    tmp2 = tmp_plugin.spawn(register=True)

    @tmp_plugin.contribute.reader(filename_patterns=['*.fake'])
    def read_tif(path):
        ...

    @tmp2.contribute.reader(filename_patterns=['.fake2'])
    def read_all(path):
        ...

    npe2_readers, npe1_readers = get_all_readers()
    assert len(npe2_readers) == 2
    assert len(npe1_readers) == 0


def test_get_filename_patterns_fake_plugin():
    assert len(get_filename_patterns_for_reader('gibberish')) == 0


def test_get_filename_patterns(tmp_plugin: DynamicPlugin):
    @tmp_plugin.contribute.reader(filename_patterns=['*.tif'])
    def read_tif(path):
        ...

    @tmp_plugin.contribute.reader(filename_patterns=['*.csv'])
    def read_csv(pth):
        ...

    patterns = get_filename_patterns_for_reader(tmp_plugin.name)
    assert len(patterns) == 2
    assert '*.tif' in patterns
    assert '*.csv' in patterns
