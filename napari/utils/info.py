import os
import platform
import subprocess
import sys

import napari

OS_RELEASE_PATH = "/etc/os-release"


def _linux_sys_name():
    """
    Try to discover linux system name base on /etc/os-release file or lsb_release command output
    https://www.freedesktop.org/software/systemd/man/os-release.html
    """
    if os.path.exists(OS_RELEASE_PATH):
        with open(OS_RELEASE_PATH) as f_p:
            data = {}
            for line in f_p:
                field, value = line.split("=")
                data[field.strip()] = value.strip().strip('"')
        if "PRETTY_NAME" in data:
            return data["PRETTY_NAME"]
        if "NAME" in data:
            if "VERSION" in data:
                return f'{data["NAME"]} {data["VERSION"]}'
            if "VERSION_ID" in data:
                return f'{data["NAME"]} {data["VERSION_ID"]}'
            return f'{data["NAME"]} (no version)'

    try:
        res = subprocess.run(
            ["lsb_release", "-d", "-r"], check=True, capture_output=True
        )
        text = res.stdout.decode()
        data = {}
        for line in text.split("\n"):
            key, val = line.split(":")
            data[key.strip()] = val.strip()
        version_str = data["Description"]
        if not version_str.endswith(data["Release"]):
            version_str += " " + data["Release"]
        return version_str
    except subprocess.CalledProcessError:
        pass
    return ""


def _sys_name():
    """
    Discover MacOS or Linux Human readable information. For Linux provide information about distribution.
    """
    try:
        if sys.platform == "linux":
            return _linux_sys_name()
        if sys.platform == "darwin":
            try:
                res = subprocess.run(
                    ["sw_vers", "-productVersion"],
                    check=True,
                    capture_output=True,
                )
                return f"MacOS {res.stdout.decode().strip()}"
            except subprocess.CalledProcessError:
                pass
    except Exception:
        pass
    return ""


def sys_info(as_html=False):
    """Gathers relevant module versions for troubleshooting purposes.

    Parameters
    ----------
    as_html : bool
        if True, info will be returned as HTML, suitable for a QTextEdit widget
    """
    sys_version = sys.version.replace('\n', ' ')
    text = (
        f"<b>napari</b>: {napari.__version__}<br>"
        f"<b>Platform</b>: {platform.platform()}<br>"
    )

    __sys_name = _sys_name()
    if __sys_name:
        text += f"<b>System</b>: {__sys_name}<br>"

    text += f"<b>Python</b>: {sys_version}<br>"

    try:
        from qtpy import API_NAME, PYQT_VERSION, PYSIDE_VERSION, QtCore

        if API_NAME == 'PySide2':
            API_VERSION = PYSIDE_VERSION
        elif API_NAME == 'PyQt5':
            API_VERSION = PYQT_VERSION
        else:
            API_VERSION = ''

        text += (
            f"<b>Qt</b>: {QtCore.__version__}<br>"
            f"<b>{API_NAME}</b>: {API_VERSION}<br>"
        )

    except Exception as e:
        text += f"<b>Qt</b>: Import failed ({e})<br>"

    modules = (
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('dask', 'Dask'),
        ('vispy', 'VisPy'),
    )

    loaded = {}
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f"<b>{name}</b>: {loaded[module].__version__}<br>"
        except Exception as e:
            text += f"<b>{name}</b>: Import failed ({e})<br>"

    text += "<br><b>OpenGL:</b><br>"

    if loaded.get('vispy', False):
        sys_info_text = (
            "<br>".join(
                [
                    loaded['vispy'].sys_info().split("\n")[index]
                    for index in [-4, -3]
                ]
            )
            .replace("'", "")
            .replace("<br>", "<br>  - ")
        )
        text += f'  - {sys_info_text}<br>'
    else:
        text += "  - failed to load vispy"

    text += "<br><b>Screens:</b><br>"

    try:
        from qtpy.QtGui import QGuiApplication

        screen_list = QGuiApplication.screens()
        for i, screen in enumerate(screen_list, start=1):
            text += f"  - screen {i}: resolution {screen.geometry().width()}x{screen.geometry().height()}, scale {screen.devicePixelRatio()}<br>"
    except Exception as e:
        text += f"  - failed to load screen information {e}"

    if not as_html:
        text = (
            text.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
        )
    return text


citation_text = (
    'napari contributors (2019). napari: a '
    'multi-dimensional image viewer for python. '
    'doi:10.5281/zenodo.3555620'
)
