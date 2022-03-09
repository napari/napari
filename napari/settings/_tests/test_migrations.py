from importlib.metadata import distribution
from unittest.mock import Mock, patch

import pytest

from napari.settings import NapariSettings, _migrations


@pytest.fixture
def _test_migrator(monkeypatch):
    _TEST_MIGRATORS = []
    with monkeypatch.context() as m:
        m.setattr(_migrations, "_MIGRATORS", _TEST_MIGRATORS)
        yield _migrations.migrator


def test_no_migrations_available(_test_migrator):
    settings = NapariSettings(schema_version='0.1.0')
    assert settings.schema_version == '0.1.0'


def test_empty_migration_changes_version(_test_migrator):
    mock = Mock()

    @_test_migrator('0.1.0', '0.2.0')
    def _(model):
        mock()

    settings = NapariSettings(schema_version='0.1.0')
    assert settings.schema_version == '0.2.0'
    mock.assert_called_once()


def test_failed_migration_leaves_version(_test_migrator):
    @_test_migrator('0.1.0', '0.2.0')
    def _(model: NapariSettings):
        model.appearance.theme = 'light'
        assert model.appearance.theme == 'light'
        raise ValueError('broken migration')

    with pytest.warns(UserWarning) as e:
        settings = NapariSettings(schema_version='0.1.0')
    assert settings.schema_version == '0.1.0'
    # test migration was atomic, and reverted the theme change
    assert settings.appearance.theme == 'dark'
    assert 'Failed to migrate settings from v0.1.0 to v0.2.0' in str(e[0])


@patch(
    'napari.settings._migrations.do_migrations',
    wraps=_migrations.do_migrations,
)
def test_napari_svg_unblocked(mock):
    try:
        d = distribution('napari-svg')
        assert 'napari.manifest' in {ep.group for ep in d.entry_points}
    except Exception:
        pytest.xfail('napari-svg not present as an npe2 plugin')

    settings = NapariSettings(
        None,
        schema_version='0.3.0',
        plugins={'disabled_plugins': {'napari-svg'}},
    )
    mock.assert_called_once()
    assert 'napari-svg' not in settings.plugins.disabled_plugins
