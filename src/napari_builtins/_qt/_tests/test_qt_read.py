"""Qt-dependent integration tests for io/_read.py"""

import pytest

from napari.utils.io import _SCRIPT_NAMESPACES
from napari_builtins.io._read import load_and_execute_python_code


class TestLoadAndExecutePythonCode:
    """execute_python_code imports napari._qt (via _noop_napari_run), so any test
    that calls load_and_execute_python_code end-to-end requires a Qt binding.
    These tests verify that a file written in a specific encoding is both decoded
    correctly *and* executed so that the resulting namespace is correct.
    The pure decoding step (_read_python_source) is tested
    headlessly in src/napari_builtins/_tests/test_read.py.
    """

    @pytest.mark.usefixtures('qapp')
    @pytest.mark.parametrize(
        ('encoding', 'prefix', 'unit_value'),
        [
            ('utf-8', '', 'μm'),
            ('latin-1', '# coding: latin-1\n', 'µm'),
        ],
    )
    def test_load_and_execute_python_code_uses_python_source_encoding(
        self, tmp_path, encoding, prefix, unit_value
    ):
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
