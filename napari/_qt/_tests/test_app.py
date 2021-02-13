import os

import pytest

from napari._qt.qt_event_loop import set_app_id


@pytest.mark.skipif(os.name != "Windows", reason="Windows specific")
def test_windows_grouping_overwrite(make_napari_viewer):
    import ctypes

    def get_app_id():
        mem = ctypes.POINTER(ctypes.c_wchar)()
        ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(
            ctypes.byref(mem)
        )
        res = ctypes.wstring_at(mem)
        ctypes.windll.Ole32.CoTaskMemFree(mem)
        return res

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("test_text")

    assert get_app_id() == "test_text"
    set_app_id("custom_string")
    assert get_app_id() == "custom_string"
    set_app_id("")
    assert get_app_id() == ""
