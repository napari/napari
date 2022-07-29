from npe2 import DynamicPlugin

from napari.plugins.utils import (
    MatchFlag,
    get_all_readers,
    get_filename_patterns_for_reader,
    get_potential_readers,
    get_preferred_reader,
    score_specificity,
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
        '*/my-specific-folder/*.tif': 'fake-plugin'
    }
    reader = get_preferred_reader('/asdf/my-specific-folder/my_file.tif')
    assert reader == 'fake-plugin'

    reader = get_preferred_reader('/asdf/foo/my-specific-folder/my_file.tif')
    assert reader == 'fake-plugin'


def test_get_preferred_reader_match_less_ambiguous():
    get_settings().plugins.extension2reader = {
        # generic star so least specificity
        '*.tif': 'generic-tif-plugin',
        # specific file so most specificity
        '*/foo.tif': 'very-specific-plugin',
        # set so less specificity
        '*/file_[0-9][0-9].tif': 'set-plugin',
    }

    reader = get_preferred_reader('/asdf/a.tif')
    assert reader == 'generic-tif-plugin'

    reader = get_preferred_reader('/asdf/foo.tif')
    assert reader == 'very-specific-plugin'

    reader = get_preferred_reader('/asdf/file_01.tif')
    assert reader == 'set-plugin'


def test_get_preferred_reader_more_nested():
    get_settings().plugins.extension2reader = {
        # less nested so less specificity
        '*.tif': 'generic-tif-plugin',
        # more nested so higher specificity
        '*/my-specific-folder/*.tif': 'fake-plugin',
        # even more nested so even higher specificity
        '*/my-specific-folder/nested/*.tif': 'very-specific-plugin',
    }

    reader = get_preferred_reader('/asdf/nested/1/2/3/my_file.tif')
    assert reader == 'generic-tif-plugin'

    reader = get_preferred_reader('/asdf/my-specific-folder/my_file.tif')
    assert reader == 'fake-plugin'

    reader = get_preferred_reader(
        '/asdf/my-specific-folder/nested/my_file.tif'
    )
    assert reader == 'very-specific-plugin'


def test_get_preferred_reader_abs_path():
    get_settings().plugins.extension2reader = {
        # abs path so highest specificity
        '/asdf/*.tif': 'most-specific-plugin',
        # less nested so less specificity
        '*.tif': 'generic-tif-plugin',
        # more nested so higher specificity
        '*/my-specific-folder/*.tif': 'fake-plugin',
        # even more nested so even higher specificity
        '*/my-specific-folder/nested/*.tif': 'very-specific-plugin',
    }

    reader = get_preferred_reader(
        '/asdf/my-specific-folder/nested/my_file.tif'
    )
    assert reader == 'most-specific-plugin'


def test_score_specificity_simple():
    assert score_specificity('') == (True, 0, [MatchFlag.NONE])
    assert score_specificity('a') == (True, 0, [MatchFlag.NONE])
    assert score_specificity('ab*c') == (True, 0, [MatchFlag.STAR])
    assert score_specificity('a?c') == (True, 0, [MatchFlag.ANY])
    assert score_specificity('a[a-zA-Z]c') == (True, 0, [MatchFlag.SET])
    assert score_specificity('*[a-zA-Z]*a?c') == (
        True,
        0,
        [MatchFlag.STAR | MatchFlag.ANY | MatchFlag.SET],
    )


def test_score_specificity_complex():
    assert score_specificity('*/my-specific-folder/[nested]/*?.tif') == (
        True,
        -3,
        [
            MatchFlag.STAR,
            MatchFlag.NONE,
            MatchFlag.SET,
            MatchFlag.STAR | MatchFlag.ANY,
        ],
    )

    assert score_specificity('/my-specific-folder/[nested]/*?.tif') == (
        False,
        -2,
        [
            MatchFlag.NONE,
            MatchFlag.SET,
            MatchFlag.STAR | MatchFlag.ANY,
        ],
    )


def test_score_specificity_collapse_star():
    assert score_specificity('*/*/?*.tif') == (
        True,
        -1,
        [MatchFlag.STAR, MatchFlag.STAR | MatchFlag.ANY],
    )
    assert score_specificity('*/*/*a?c.tif') == (
        True,
        0,
        [MatchFlag.STAR | MatchFlag.ANY],
    )
    assert score_specificity('*/*/*.tif') == (True, 0, [MatchFlag.STAR])
    assert score_specificity('*/abc*/*.tif') == (
        True,
        -1,
        [MatchFlag.STAR, MatchFlag.STAR],
    )
    assert score_specificity('/abc*/*.tif') == (False, 0, [MatchFlag.STAR])


def test_score_specificity_range():
    _, _, score = score_specificity('[abc')
    assert score == [MatchFlag.NONE]

    _, _, score = score_specificity('[abc]')
    assert score == [MatchFlag.SET]

    _, _, score = score_specificity('[abc[')
    assert score == [MatchFlag.NONE]

    _, _, score = score_specificity('][abc')
    assert score == [MatchFlag.NONE]

    _, _, score = score_specificity('[[abc]]')
    assert score == [MatchFlag.SET]


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
