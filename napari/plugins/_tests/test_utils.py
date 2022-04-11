from napari._tests.utils import restore_settings_on_exit
from napari.plugins.utils import get_potential_readers, get_preferred_reader
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
        get_settings().plugins.extension2reader = {'.tif': 'fake-plugin'}
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
