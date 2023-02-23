import os
from importlib.metadata import PackageNotFoundError, distribution
from unittest.mock import patch

import pytest

from napari.settings import NapariSettings, _migrations


@pytest.fixture
def _test_migrator(monkeypatch):
    # this fixture makes sure we're not using _migrations.MIGRATORS for tests
    # but rather only using migrators that get declared IN the test
    _TEST_MIGRATORS = []
    with monkeypatch.context() as m:
        m.setattr(_migrations, "_MIGRATORS", _TEST_MIGRATORS)
        yield _migrations.migrator


def test_no_migrations_available(_test_migrator):
    # no migrators exist... nothing should happen
    settings = NapariSettings(schema_version='0.1.0')
    assert settings.schema_version == '0.1.0'


def test_backwards_migrator(_test_migrator):
    # we shouldn't be able to downgrade the schema version
    # if that is needed later, we can create a new decorator,
    # or change this test
    with pytest.raises(AssertionError):

        @_test_migrator('0.2.0', '0.1.0')
        def _(model):
            ...


def test_migration_works(_test_migrator):
    # test that a basic migrator works to change the version
    # and mutate the model

    @_test_migrator('0.1.0', '0.2.0')
    def _(model: NapariSettings):
        model.appearance.theme = 'light'

    settings = NapariSettings(schema_version='0.1.0')
    assert settings.schema_version == '0.2.0'
    assert settings.appearance.theme == 'light'


def test_migration_saves(_test_migrator):
    @_test_migrator('0.1.0', '0.2.0')
    def _(model: NapariSettings):
        ...

    with patch.object(NapariSettings, 'save') as mock:
        mock.assert_not_called()
        settings = NapariSettings(config_path='junk', schema_version='0.1.0')
        assert settings.schema_version == '0.2.0'
        mock.assert_called()


def test_failed_migration_leaves_version(_test_migrator):
    # if an error occurs IN the migrator, the version should stay
    # where it was before the migration, and any changes reverted.
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
    # test that the user was warned
    assert 'Failed to migrate settings from v0.1.0 to v0.2.0' in str(e[0])


@pytest.mark.skipif(
    bool(os.environ.get('MIN_REQ')), reason='not relevant for MIN_REQ'
)
def test_030_to_040_migration():
    # Prior to v0.4.0, npe2 plugins were automatically "disabled"
    # 0.3.0 -> 0.4.0 should remove any installed npe2 plugins from the
    # set of disabled plugins (see migrator for details)
    try:
        d = distribution('napari-svg')
        assert 'napari.manifest' in {ep.group for ep in d.entry_points}
    except PackageNotFoundError:
        pytest.fail(
            'napari-svg not present as an npe2 plugin. '
            'This test needs updating'
        )

    settings = NapariSettings(
        schema_version='0.3.0',
        plugins={'disabled_plugins': {'napari-svg', 'napari'}},
    )
    assert 'napari-svg' not in settings.plugins.disabled_plugins
    assert 'napari' not in settings.plugins.disabled_plugins


@pytest.mark.skipif(
    bool(os.environ.get('MIN_REQ')), reason='not relevant for MIN_REQ'
)
def test_040_to_050_migration():
    # Prior to 0.5.0 existing preferences may have reader extensions
    # preferences saved without a leading *.
    # fnmatch would fail on these so we coerce them to include a *
    # e.g. '.csv' becomes '*.csv'
    settings = NapariSettings(
        schema_version='0.4.0',
        plugins={'extension2reader': {'.tif': 'napari'}},
    )
    assert '.tif' not in settings.plugins.extension2reader
    assert '*.tif' in settings.plugins.extension2reader
