import platform
import sys

import napari


def sys_info(as_html=False):
    """Gathers relevant module versions for troubleshooting purposes.

    Parameters
    ----------
    as_html : bool
        if True, info will be returned as HTML, suitable for a QTextEdit widget
    """
    from napari.plugins import plugin_manager

    sys_version = sys.version.replace('\n', ' ')
    text = (
        f"<b>napari</b>: {napari.__version__}<br>"
        f"<b>Platform</b>: {platform.platform()}<br>"
        f"<b>Python</b>: {sys_version}<br>"
    )

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
            text += f"  - screen #{i}: resolution {screen.geometry().width()}x{screen.geometry().height()}, scale {screen.devicePixelRatio()}<br>"
    except Exception as e:
        text += f"  - failed to load screen information {e}"

    plugin_manager.discover()
    plugin_strings = []
    for meta in plugin_manager.list_plugin_metadata():
        plugin_name = meta.get('plugin_name')
        if plugin_name == 'builtins':
            continue
        version = meta.get('version')
        version_string = f": {version}" if version else ""
        plugin_strings.append(f"  - {plugin_name}{version_string}")
    text += '<br><b>Plugins</b>:'
    text += (
        ("<br>" + "<br>".join(sorted(plugin_strings)))
        if plugin_strings
        else '  None'
    )

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
