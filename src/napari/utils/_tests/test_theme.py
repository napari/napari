import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from npe2 import PluginManager, PluginManifest, __version__ as npe2_version
from npe2.manifest.schema import ContributionPoints
from packaging.version import parse as parse_version
from pydantic import ValidationError

import napari.resources._icons as icons_module
from napari.resources._icons import (
    CACHE_FINGERPRINT_FILE,
    PLUGIN_FILE_NAME,
    _get_expected_svg_filenames,
    build_theme_svgs,
)
from napari.settings import get_settings
from napari.utils.theme import (
    Theme,
    _install_npe2_themes,
    available_themes,
    get_theme,
    is_theme_available,
    register_theme,
    unregister_theme,
)


def test_default_themes():
    themes = available_themes()
    assert 'dark' in themes
    assert 'light' in themes
    assert 'system' in themes


def test_get_theme():
    # get theme in the old-style dict format
    theme = get_theme('dark').to_rgb_dict()
    assert isinstance(theme, dict)

    # get theme in the new model-based format
    theme = get_theme('dark')
    assert isinstance(theme, Theme)


def test_get_system_theme(monkeypatch):
    monkeypatch.setattr('napari.utils.theme.get_system_theme', lambda: 'light')
    theme = get_theme('system')
    # should return the theme specified by get_system_theme
    assert theme.id == 'light'


def test_register_theme():
    # Check that blue theme is not listed in available themes
    themes = available_themes()
    assert 'test_blue' not in themes

    # Create new blue theme based on dark theme
    blue_theme = get_theme('dark').to_rgb_dict()
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )

    # Register blue theme
    register_theme('test_blue', blue_theme, 'test')

    # Check that blue theme is listed in available themes
    themes = available_themes()
    assert 'test_blue' in themes

    # Check that the dark theme has not been overwritten
    dark_theme = get_theme('dark').to_rgb_dict()
    assert dark_theme['background'] != blue_theme['background']

    # Check that blue theme can be gotten from available themes
    theme = get_theme('test_blue').to_rgb_dict()
    assert theme['background'] == blue_theme['background']

    theme = get_theme('test_blue')
    assert theme.background.as_rgb() == blue_theme['background']


def test_unregister_theme():
    # Create new blue theme based on dark theme
    blue_theme = get_theme('dark').to_rgb_dict()
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )

    # Register blue theme
    register_theme('test_blue', blue_theme, 'test')

    # Check that blue theme is listed in available themes
    themes = available_themes()
    assert 'test_blue' in themes

    # Remove theme from available themes
    unregister_theme('test_blue')
    themes = available_themes()
    assert 'test_blue' not in themes


def test_rebuild_theme_settings():
    settings = get_settings()
    assert 'another-theme' not in available_themes()
    # theme is not updated
    with pytest.raises(ValidationError):
        settings.appearance.theme = 'another-theme'
    blue_theme = get_theme('dark').to_rgb_dict()
    register_theme('another-theme', blue_theme, 'test')
    settings.appearance.theme = 'another-theme'


@pytest.mark.parametrize(
    'color',
    [
        '#FF0000',
        'white',
        (0, 127, 127),
        (0, 255, 255, 0.5),
        [50, 200, 200],
        [140, 140, 140, 0.7],
    ],
)
def test_theme(color):
    theme = get_theme('dark')
    theme.background = color


def test_theme_font_size():
    theme = get_theme('dark')
    theme.font_size = '15pt'
    assert theme.font_size == '15pt'

    with pytest.raises(ValidationError):
        theme.font_size = '0pt'
    with pytest.raises(ValidationError):
        theme.font_size = '12px'


def test_theme_syntax_highlight():
    theme = get_theme('dark')
    with pytest.raises(ValidationError):
        theme.syntax_style = 'invalid'


def test_is_theme_available(tmp_path, monkeypatch):
    (tmp_path / 'test_blue').mkdir()
    (tmp_path / 'yellow').mkdir()
    (tmp_path / 'test_blue' / PLUGIN_FILE_NAME).write_text('test-blue')
    monkeypatch.setattr(
        'napari.utils.theme._theme_path', lambda x: tmp_path / x
    )

    n_themes = len(available_themes())

    def mock_install_theme(_themes):
        theme_dict = _themes['dark'].to_rgb_dict()
        theme_dict['id'] = 'test_blue'
        register_theme('test_blue', theme_dict, 'test')

    monkeypatch.setattr(
        'napari.utils.theme._install_npe2_themes', mock_install_theme
    )

    assert len(available_themes()) == n_themes
    assert is_theme_available('dark')
    assert not is_theme_available('green')
    assert not is_theme_available('yellow')
    assert is_theme_available('test_blue')
    assert len(available_themes()) == n_themes + 1
    assert 'test_blue' in available_themes()


@pytest.mark.skipif(
    parse_version(npe2_version) < parse_version('0.6.2'),
    reason='requires npe2 0.6.2 for syntax style contributions',
)
def test_theme_registration(monkeypatch, caplog):
    data = {'dark': get_theme('dark')}

    manifest = PluginManifest(
        name='theme_test',
        display_name='Theme Test',
        contributions=ContributionPoints(
            themes=[
                {
                    'id': 'theme1',
                    'label': 'Theme 1',
                    'type': 'dark',
                    'syntax_style': 'native',
                    'colors': {},
                },
                {
                    'id': 'theme2',
                    'label': 'Theme 2',
                    'type': 'dark',
                    'syntax_style': 'does_not_exist',
                    'colors': {},
                },
            ]
        ),
    )

    def mock_iter_manifests(disabled):
        return [manifest]

    monkeypatch.setattr(
        PluginManager.instance(), 'iter_manifests', mock_iter_manifests
    )
    monkeypatch.setattr('napari.utils.theme._themes', data)
    _install_npe2_themes(data)

    assert 'theme1' in data
    assert 'theme2' not in data
    assert 'Registration theme failed' in caplog.text


def test_build_theme_svgs_initial_creation_and_cache_hit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 1
    assert out_dir.exists()

    fingerprint_file = out_dir / CACHE_FINGERPRINT_FILE
    assert fingerprint_file.is_file()
    initial_fingerprint = fingerprint_file.read_text(encoding='utf-8').strip()
    assert len(initial_fingerprint) == 64  # SHA-256 hexdigest

    expected_files = _get_expected_svg_filenames()
    for fname in expected_files:
        assert (out_dir / fname).is_file()
    assert (out_dir / PLUGIN_FILE_NAME).read_text(
        encoding='utf-8'
    ) == 'builtin'

    # Second call with the same theme must skip generation
    out_dir_2 = Path(build_theme_svgs('dark', 'builtin'))
    assert str(out_dir_2) == str(out_dir)
    assert spy.call_count == 1  # Not called again!

    # Calling with different source updates plugin.txt without rebuilding SVGs
    build_theme_svgs('dark', 'custom_plugin')
    assert spy.call_count == 1
    assert (out_dir / PLUGIN_FILE_NAME).read_text(
        encoding='utf-8'
    ) == 'custom_plugin'


def test_build_theme_svgs_rebuild_on_color_change(tmp_path, monkeypatch):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)

    custom_theme = get_theme('dark').to_rgb_dict()
    custom_theme['id'] = 'test_color_cache'

    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    try:
        register_theme('test_color_cache', custom_theme, 'test')
        assert spy.call_count == 1
        out_dir = tmp_path / 'test_color_cache'
        initial_fp = (
            (out_dir / CACHE_FINGERPRINT_FILE)
            .read_text(encoding='utf-8')
            .strip()
        )

        # Calling build_theme_svgs again without changes should be a cache hit
        build_theme_svgs('test_color_cache', 'test')
        assert spy.call_count == 1

        # Changing an irrelevant color (e.g. background) should NOT invalidate the SVG cache
        updated_theme_bg = get_theme('test_color_cache').to_rgb_dict()
        updated_theme_bg['background'] = 'rgb(10, 20, 30)'
        register_theme('test_color_cache', updated_theme_bg, 'test')
        assert spy.call_count == 1
        assert (out_dir / CACHE_FINGERPRINT_FILE).read_text(
            encoding='utf-8'
        ).strip() == initial_fp

        # Update a relevant color ('icon')
        updated_theme_icon = get_theme('test_color_cache').to_rgb_dict()
        updated_theme_icon['icon'] = 'rgb(123, 45, 67)'
        register_theme('test_color_cache', updated_theme_icon, 'test')

        # build_theme_svgs was invoked by register_theme, so spy count increased
        assert spy.call_count == 2
        new_fp = (
            (out_dir / CACHE_FINGERPRINT_FILE)
            .read_text(encoding='utf-8')
            .strip()
        )
        assert new_fp != initial_fp
    finally:
        unregister_theme('test_color_cache')


def test_build_theme_svgs_rebuild_on_source_svg_change(tmp_path, monkeypatch):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 1

    # Simulate source SVG change by mocking source SVG hash
    monkeypatch.setattr(
        icons_module,
        '_compute_source_svgs_hash',
        lambda icons=None: 'mocked_different_source_hash',
    )

    Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 2


def test_build_theme_svgs_rebuild_on_missing_output_svg(tmp_path, monkeypatch):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 1

    # Delete one expected SVG file to simulate incomplete cache
    missing_file = out_dir / 'check.svg'
    assert missing_file.is_file()
    missing_file.unlink()
    assert not missing_file.exists()

    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 2
    assert missing_file.is_file()


def test_build_theme_svgs_rebuild_on_missing_or_corrupt_fingerprint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 1
    fp_file = out_dir / CACHE_FINGERPRINT_FILE
    assert fp_file.is_file()

    # Case 1: Missing fingerprint file
    fp_file.unlink()
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 2
    assert fp_file.is_file()

    # Case 2: Corrupted fingerprint file
    fp_file.write_text('corrupt_fingerprint_data', encoding='utf-8')
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 3
    assert (
        fp_file.read_text(encoding='utf-8').strip()
        != 'corrupt_fingerprint_data'
    )


def test_build_theme_svgs_interrupted_write_clears_fingerprint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    fp_file = out_dir / CACHE_FINGERPRINT_FILE
    assert fp_file.is_file()

    # Invalidate cache so a rebuild is required
    fp_file.write_text('stale_fingerprint', encoding='utf-8')

    # Simulate write failure during rebuild
    def fail_write(*args, **kwargs):
        raise RuntimeError('Disk full simulation')

    monkeypatch.setattr(icons_module, 'write_colorized_svgs', fail_write)

    with pytest.raises(RuntimeError, match='Disk full simulation'):
        out_dir = Path(build_theme_svgs('dark', 'builtin'))

    # The stale fingerprint must have been removed so it cannot validate partial state
    assert not fp_file.exists()


def test_register_theme_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    monkeypatch.setattr(
        'napari.utils.theme._theme_path', lambda x: tmp_path / x
    )

    custom_theme = get_theme('dark').to_rgb_dict()
    custom_theme['id'] = 'test_reg_cache'

    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    try:
        register_theme('test_reg_cache', custom_theme, 'test')
        assert spy.call_count == 1

        # Re-registering the same theme must skip SVG writing
        register_theme('test_reg_cache', custom_theme, 'test')
        assert spy.call_count == 1
    finally:
        unregister_theme('test_reg_cache')


def test_compute_source_svgs_hash_with_dict_and_detects_changes(tmp_path):
    svg1 = tmp_path / 'icon1.svg'
    svg1.write_text('<svg>1</svg>')
    icons_dict = {'icon1': svg1}

    hash1 = icons_module._compute_source_svgs_hash(icons_dict)
    assert len(hash1) == 64

    # Modifying content of svg1 changes hash
    svg1.write_text('<svg>2</svg>')
    hash2 = icons_module._compute_source_svgs_hash(icons_dict)
    assert hash2 != hash1

    # Missing/unreadable file raises OSError (not silently swallowed)
    svg1.unlink()
    with pytest.raises(FileNotFoundError):
        icons_module._compute_source_svgs_hash(icons_dict)


def test_write_fingerprint_atomic_cleanup_on_error(tmp_path, monkeypatch):
    dest_dir = tmp_path / 'atomic_test'
    dest_dir.mkdir()

    monkeypatch.setattr(
        icons_module.os,
        'replace',
        MagicMock(side_effect=OSError('Simulated replace failure')),
    )

    with pytest.raises(OSError, match='Simulated replace failure'):
        icons_module._write_fingerprint_atomic(dest_dir, 'some_fingerprint')

    # Ensure no leftover temporary files in dest_dir
    assert list(dest_dir.iterdir()) == []


def test_remove_fingerprint_file_raises_on_error(tmp_path, monkeypatch):
    fp_file = tmp_path / icons_module.CACHE_FINGERPRINT_FILE
    fp_file.write_text('fp')

    def fail_unlink(path, *args, **kwargs):
        raise PermissionError('Simulated locked file')

    monkeypatch.setattr(Path, 'unlink', fail_unlink)

    with pytest.raises(PermissionError, match='Simulated locked file'):
        icons_module._remove_fingerprint_file(tmp_path)


def test_compute_source_svgs_hash_detects_content_change_with_same_size_and_mtime(
    tmp_path,
):
    svg1 = tmp_path / 'icon1.svg'
    svg1.write_text('<svg>A</svg>', encoding='utf-8')
    stat1 = svg1.stat()
    icons_dict = {'icon1': svg1}

    hash1 = icons_module._compute_source_svgs_hash(icons_dict)

    # Modify content without changing file size, and restore exact mtime_ns
    svg1.write_text('<svg>B</svg>', encoding='utf-8')
    os.utime(svg1, ns=(stat1.st_atime_ns, stat1.st_mtime_ns))
    stat2 = svg1.stat()
    assert stat2.st_size == stat1.st_size
    assert stat2.st_mtime_ns == stat1.st_mtime_ns

    # Even with identical size and mtime_ns, SHA-256 of content detects the change
    hash2 = icons_module._compute_source_svgs_hash(icons_dict)
    assert hash2 != hash1


def test_build_theme_svgs_rebuild_on_empty_plugin_txt(tmp_path, monkeypatch):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    plugin_file = out_dir / PLUGIN_FILE_NAME
    assert plugin_file.read_text(encoding='utf-8').strip() == 'builtin'

    # Empty plugin.txt
    plugin_file.write_text('', encoding='utf-8')
    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    # Must detect invalid cache and rebuild
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 1
    assert plugin_file.read_text(encoding='utf-8').strip() == 'builtin'


def test_build_theme_svgs_rebuild_updates_generated_svg_when_source_changes_in_process(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / 'cache'
    icons_dir = tmp_path / 'icons'
    icons_dir.mkdir()
    cache_dir.mkdir()

    monkeypatch.setattr(icons_module, '_theme_path', lambda x: cache_dir / x)

    # 1. Create source SVG with initial content
    svg_source = icons_dir / 'my_icon.svg'
    svg_source.write_text("<svg><path id='v1'/></svg>", encoding='utf-8')

    monkeypatch.setattr(icons_module, 'ICONS', {'my_icon': str(svg_source)})

    # First build: populates cache and in-memory lru_cache with v1
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    generated_file = out_dir / 'my_icon.svg'
    assert generated_file.is_file()
    assert "id='v1'" in generated_file.read_text(encoding='utf-8')

    # 2. Modify source SVG in disk during the same process
    svg_source.write_text("<svg><path id='v2'/></svg>", encoding='utf-8')

    # Second build: must invalidate cache and generate new SVG containing v2,
    # rather than serving stale v1 from in-memory lru_cache
    out_dir_2 = Path(build_theme_svgs('dark', 'builtin'))
    generated_file_2 = out_dir_2 / 'my_icon.svg'
    content_after = generated_file_2.read_text(encoding='utf-8')
    assert "id='v2'" in content_after
    assert "id='v1'" not in content_after


def test_build_theme_svgs_rebuild_on_empty_output_svg(tmp_path, monkeypatch):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    check_svg = out_dir / 'check.svg'
    assert check_svg.is_file()
    assert check_svg.stat().st_size > 0

    # Truncate an output SVG file to 0 bytes to simulate corrupted/empty output
    check_svg.write_text('', encoding='utf-8')
    assert check_svg.stat().st_size == 0

    real_write = icons_module.write_colorized_svgs
    spy = MagicMock(side_effect=real_write)
    monkeypatch.setattr(icons_module, 'write_colorized_svgs', spy)

    # Must detect empty file as invalid cache and rebuild
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    assert spy.call_count == 1
    assert check_svg.stat().st_size > 0


def test_remove_fingerprint_file_handles_missing_file_without_toctou(tmp_path):
    # Removing a non-existent fingerprint file must not raise FileNotFoundError
    icons_module._remove_fingerprint_file(tmp_path)
    assert not (tmp_path / icons_module.CACHE_FINGERPRINT_FILE).exists()


def test_build_theme_svgs_ensures_no_stale_fingerprint_on_write_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(icons_module, '_theme_path', lambda x: tmp_path / x)
    out_dir = Path(build_theme_svgs('dark', 'builtin'))
    fp_file = out_dir / CACHE_FINGERPRINT_FILE
    assert fp_file.is_file()

    # Invalidate cache by updating source hash mock
    monkeypatch.setattr(
        icons_module,
        '_compute_source_svgs_hash',
        lambda icons=None: 'different_hash',
    )

    # Simulate write failure during colorized SVGs write
    def fail_write(*args, **kwargs):
        raise OSError('Simulated disk full')

    monkeypatch.setattr(icons_module, 'write_colorized_svgs', fail_write)

    with pytest.raises(OSError, match='Simulated disk full'):
        out_dir = Path(build_theme_svgs('dark', 'builtin'))

    # Fingerprint must be removed and not present
    assert not fp_file.exists()
