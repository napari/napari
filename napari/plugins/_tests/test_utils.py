from napari._tests.utils import restore_settings_on_exit
from napari.plugins.utils import (
    get_all_readers,
    get_filename_patterns_for_reader,
    get_potential_readers,
    get_preferred_reader,
)
from napari.settings import get_settings


def test_get_preferred_reader_no_readers():
    pth = 'my_file.tif'
    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {}
        reader = get_preferred_reader(pth)
        assert reader is None


def test_get_preferred_reader_for_extension():
    pth = 'my_file.tif'
    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {'*.tif': 'fake-plugin'}
        reader = get_preferred_reader(pth)
        assert reader == 'fake-plugin'


def test_get_preferred_reader_complex_pattern():
    pth = 'my-specific-folder/my_file.tif'
    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {
            'my-specific-folder/*.tif': 'fake-plugin'
        }
        reader = get_preferred_reader(pth)
        assert reader == 'fake-plugin'


def test_get_preferred_reader_no_extension():
    pth = 'my_file'
    reader = get_preferred_reader(pth)
    assert reader is None


def test_get_potential_readers_finds_npe1(mock_npe2_pm):
    pth = 'my_file.tif'
    readers = get_potential_readers(pth)
    assert 'builtins' in readers


def test_get_potential_readers_gives_napari(mock_npe2_pm, tmp_reader):
    pth = 'my_file.tif'

    tmp_reader(mock_npe2_pm, 'napari', ['*.tif'])
    readers = get_potential_readers(pth)
    assert 'napari' in readers
    assert 'builtins' not in readers


def test_get_potential_readers_finds_readers(mock_npe2_pm, tmp_reader):
    pth = 'my_file.tif'

    tmp_reader(mock_npe2_pm, 'tif-reader', ['*.tif'])
    tmp_reader(mock_npe2_pm, 'all-reader', ['*.*'])

    readers = get_potential_readers(pth)
    assert len(readers) == 3


def test_get_potential_readers_none_available(mock_npe2_pm):
    pth = 'my_file.fake'

    readers = get_potential_readers(pth)
    assert len(readers) == 0


def test_get_potential_readers_plugin_name_disp_name(mock_npe2_pm, tmp_reader):
    pth = 'my_file.fake'

    fake_reader = tmp_reader(mock_npe2_pm, 'fake-reader')
    fake_reader.manifest.display_name = 'Fake Reader'
    readers = get_potential_readers(pth)

    assert readers['fake-reader'] == 'Fake Reader'


def test_get_all_readers_gives_npe1(mock_npe2_pm):
    """When there's no npe2 files, get_all_readers returns npe1 builtins"""
    npe2_readers, npe1_readers = get_all_readers()
    assert len(npe2_readers) == 0
    assert 'builtins' in npe1_readers


def test_get_all_readers_gives_napari():
    npe2_readers, npe1_readers = get_all_readers()
    assert len(npe1_readers) == 0
    assert len(npe2_readers) == 1
    assert 'napari' in npe2_readers


def test_get_all_readers(mock_npe2_pm, tmp_reader):
    tmp_reader(mock_npe2_pm, 'reader-1')
    tmp_reader(mock_npe2_pm, 'reader-2')
    npe2_readers, npe1_readers = get_all_readers()
    assert len(npe2_readers) == 2
    assert len(npe1_readers) == 1


def test_get_filename_patterns_fake_plugin():
    assert len(get_filename_patterns_for_reader('gibberish')) == 0


def test_get_filename_patterns(mock_npe2_pm, tmp_reader):
    fake_reader = tmp_reader(mock_npe2_pm, 'fake-reader', ['*.tif'])

    @fake_reader.contribute.reader(filename_patterns=['*.csv'])
    def read_func(pth):
        ...

    patterns = get_filename_patterns_for_reader('fake-reader')
    assert len(patterns) == 2
    assert '*.tif' in patterns
    assert '*.csv' in patterns
