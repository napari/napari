import os
import sys

import pytest
import qtpy
from qtpy.QtWidgets import QApplication

from napari._qt.qt_event_loop import __version__, get_app, set_app_id
from napari._qt.utils import delete_qapp

LINUX_CI_PYSIDE = bool(
    sys.platform.startswith('linux')
    and os.getenv('CI', '0') != '0'
    and qtpy.API_NAME == "PySide2"
)


custom = {
    'app_name': 'custom',
    'app_version': 'custom',
    'icon': 'custom',
    'org_name': 'custom',
    'org_domain': 'custom.com',
}


@pytest.fixture
def qapp(request):
    if QApplication.instance():
        delete_qapp(QApplication.instance())
    yield get_app(**request.getfixturevalue('app_kwargs'))


# @pytest.mark.skipif(LINUX_CI_PYSIDE, "Can't recreate pyside QApp on CI linux")
@pytest.mark.parametrize('app_kwargs', [custom, {}], ids=['custom', 'default'])
def test_get_app(qtbot, app_kwargs):
    """Test that calling get_app defines the attributes of the QApp."""
    app = None

    def _assert():
        assert app.applicationName() == app_kwargs.get("app_name", 'napari')
        assert app.applicationVersion() == app_kwargs.get(
            "app_version", __version__
        )
        assert app.organizationName() == app_kwargs.get("org_name", 'napari')
        assert app.organizationDomain() == app_kwargs.get(
            "org_domain", 'napari.org'
        )

    app = get_app()
    _assert()

    # QApp is a singleton, calling a second time has no effect
    app = get_app(app_name='x', app_version='x', org_name='x', org_domain='x')
    _assert()


@pytest.mark.skipif(os.name != "Windows", reason="Windows specific")
def test_windows_grouping_overwrite(make_test_viewer):
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
