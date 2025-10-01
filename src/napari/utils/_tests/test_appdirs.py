from importlib.metadata import version

from packaging.version import parse as parse_version

from napari.utils import _appdirs


def test_maybe_migrate_uvx_settings_no_uv(monkeypatch):
    monkeypatch.setattr(_appdirs, 'environment_marker', 'aaa')
    assert not _appdirs._maybe_migrate_uvx_settings()


def test_maybe_migrate_uvx_settings_no_config(monkeypatch, tmp_path):
    monkeypatch.setattr(_appdirs, 'environment_marker', 'uvx/aaa')
    monkeypatch.setattr(
        _appdirs, 'user_config_dir', lambda: str(tmp_path / 'config' / '0.1.0')
    )
    assert not _appdirs._maybe_migrate_uvx_settings()


def test_maybe_migrate_uvx_settings_config_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(_appdirs, 'environment_marker', 'uvx/aaa')
    config_path = tmp_path / 'config' / '0.1.0'
    config_path.mkdir(parents=True)
    monkeypatch.setattr(_appdirs, 'user_config_dir', lambda: str(config_path))
    assert not _appdirs._maybe_migrate_uvx_settings()


def test__maybe_migrate_uvx_settings_migrate(monkeypatch, tmp_path):
    napari_version = parse_version(version('napari')).base_version
    monkeypatch.setattr(
        _appdirs, 'environment_marker', f'uvx/{napari_version}'
    )
    base_config_path = tmp_path / 'config'
    base_config_path.mkdir(parents=True)
    (base_config_path / '0.1.0').mkdir()
    (base_config_path / '0.1.0' / 'aa.txt').write_text('aa')
    (base_config_path / '0.2.0').mkdir()
    (base_config_path / '0.2.0' / 'bb.txt').write_text('bb')
    (base_config_path / '1000.2.0').mkdir()
    (base_config_path / '1000.2.0' / 'cc.txt').write_text('cc')

    monkeypatch.setattr(
        _appdirs,
        'user_config_dir',
        lambda: str(base_config_path / napari_version),
    )
    assert _appdirs._maybe_migrate_uvx_settings()
    assert (base_config_path / napari_version).exists()
    assert (base_config_path / napari_version / 'bb.txt').exists()
