import pytest

from napari.settings import NapariSettings
from napari.settings._shortcuts import ShortcutsSettings
from napari.utils.key_bindings import KeyBinding


@pytest.mark.parametrize('bindings', [['Ctrl+U'], []])
def test_legacy_shortcuts_preserved(bindings):
    original = {'napari:reset_view': bindings}
    settings = ShortcutsSettings(shortcuts=original)

    assert settings.shortcuts['napari.reset_view'] == [
        KeyBinding.from_str(binding) for binding in bindings
    ]
    assert 'napari:reset_view' not in settings.shortcuts
    assert original == {'napari:reset_view': bindings}


@pytest.mark.parametrize('legacy_first', [True, False])
def test_dotted_shortcut_takes_precedence(legacy_first):
    items = [('napari:reset_view', ['U']), ('napari.reset_view', [])]
    if not legacy_first:
        items.reverse()
    settings = ShortcutsSettings(shortcuts=dict(items))
    assert settings.shortcuts['napari.reset_view'] == []


def test_plugin_shortcuts_normalized():
    settings = ShortcutsSettings(shortcuts={'my-plugin:run': ['U']})
    assert settings.shortcuts['my-plugin.run'] == [KeyBinding.from_str('U')]
    assert 'my-plugin:run' not in settings.shortcuts


def test_legacy_shortcuts_round_trip(tmp_path):
    settings = NapariSettings(
        shortcuts={'shortcuts': {'napari:reset_view': ['U']}}
    )
    path = tmp_path / 'settings.yaml'
    settings.save(path)
    assert 'napari:reset_view' not in path.read_text()
    reloaded = NapariSettings(config_path=path)
    assert reloaded.shortcuts.shortcuts['napari.reset_view'] == [
        KeyBinding.from_str('U')
    ]
