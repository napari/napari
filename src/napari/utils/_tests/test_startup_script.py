from types import SimpleNamespace

import pytest

from napari.utils import _startup_script


def test_run_configured_startup_script_warns_on_source_read_error(
    monkeypatch, tmp_path
):
    script_path = tmp_path / 'startup.py'
    script_path.write_text('print("hello")')

    settings = SimpleNamespace(
        application=SimpleNamespace(startup_script=script_path)
    )

    def _raise(*_args, **_kwargs):
        raise SyntaxError('bad encoding cookie')

    monkeypatch.setattr('napari.settings.get_settings', lambda: settings)
    monkeypatch.setattr(_startup_script.tokenize, 'open', _raise)
    monkeypatch.setattr(_startup_script, 'startup_script_status_info', None)

    with pytest.warns(UserWarning, match='Failed to read startup script'):
        _startup_script._run_configured_startup_script()

    assert _startup_script.startup_script_status_info is None
