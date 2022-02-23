import pytest

from napari.resources._icons import ICON_PATH, ICONS
from napari.utils.misc import dir_hash, paths_hash


def test_icon_hash_equality():
    dir_hash_result = dir_hash(ICON_PATH)
    paths_hash_result = paths_hash(ICONS.values())
    assert dir_hash_result == paths_hash_result


def test_pyside2_rcc_first():
    """
    Ensure pyside2-rcc is checked before rcc. Otherwise
    we might use an older qt's rcc with no -g support.
    """
    try:
        from napari._qt.qt_resources._icons import _find_pyside2_rcc

        exe_name = next(exe for _, exe in _find_pyside2_rcc())
        assert "pyside2-" in exe_name
    except ModuleNotFoundError as exc:
        if exc.name == "PySide2":
            pytest.xfail("Test requires PySide2")
        raise exc
