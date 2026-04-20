import pytest

from napari.utils.io import _SCRIPT_NAMESPACES
from napari.utils.notifications import notification_manager
from napari_builtins.io import _read
from napari_builtins.io._read import load_and_execute_python_code


@pytest.mark.parametrize(
    ('encoding', 'prefix', 'unit_value'),
    [
        ('utf-8', '', 'μm'),
        ('latin-1', '# coding: latin-1\n', 'µm'),
    ],
)
def test_load_and_execute_python_code_uses_python_source_encoding(
    tmp_path, make_napari_viewer, encoding, prefix, unit_value
):
    # Script execution patches the active Qt viewer, so this belongs here.
    make_napari_viewer()

    script_path = tmp_path / 'unit_script.py'
    script = f'{prefix}unit = {unit_value!r}\n'
    script_path.write_bytes(script.encode(encoding))

    key = str(script_path)
    _SCRIPT_NAMESPACES.pop(key, None)

    try:
        load_and_execute_python_code(key)
        assert _SCRIPT_NAMESPACES[key]['unit'] == unit_value
    finally:
        _SCRIPT_NAMESPACES.pop(key, None)


def test_load_and_execute_python_code_reports_source_read_errors(
    monkeypatch, make_napari_viewer
):
    # Script execution patches the active Qt viewer, so this belongs here.
    make_napari_viewer()

    error = SyntaxError('bad encoding cookie')
    captured_errors = []

    def _raise(*_args, **_kwargs):
        raise error

    def _record_error(error_type, exception, traceback):
        captured_errors.append((error_type, exception, traceback))

    monkeypatch.setattr(_read, '_read_python_source', _raise)
    monkeypatch.setattr(notification_manager, 'receive_error', _record_error)

    assert load_and_execute_python_code('broken_script.py') == [(None,)]
    assert len(captured_errors) == 1
    assert captured_errors[0][0] is SyntaxError
    assert captured_errors[0][1] is error
    assert 'broken_script.py' not in _SCRIPT_NAMESPACES
