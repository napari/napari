from napari._tests.utils import restore_settings_on_exit
from napari.settings import get_settings
from napari.settings._utils import _coerce_extensions_to_globs


def test_coercion_to_glob_deletes_existing():
    settings = {'.tif': 'fake-plugin', '*.csv': 'other-plugin'}
    settings = _coerce_extensions_to_globs(settings)
    assert '.tif' not in settings
    assert '*.tif' in settings
    assert settings['*.tif'] == 'fake-plugin'
    assert '*.csv' in settings
    assert settings['*.csv'] == 'other-plugin'


def test_coercion_to_glob_excludes_non_extensions():
    complex_pattern = '.blah*.tif'
    settings = {complex_pattern: 'fake-plugin', '*.csv': 'other-plugin'}
    settings = _coerce_extensions_to_globs(settings)
    assert '.blah*.tif' in settings
    assert settings[complex_pattern] == 'fake-plugin'


def test_coercion_to_glob_doesnt_change_settings():
    with restore_settings_on_exit():
        settings = {'*.tif': 'fake-plugin', '.csv': 'other-plugin'}
        get_settings().plugins.extension2reader = settings
        settings = _coerce_extensions_to_globs(settings)
        assert settings == {'*.tif': 'fake-plugin', '*.csv': 'other-plugin'}
        assert get_settings().plugins.extension2reader == {
            '*.tif': 'fake-plugin',
            '.csv': 'other-plugin',
        }
