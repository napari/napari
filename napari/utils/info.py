import contextlib
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import napari

OS_RELEASE_PATH = '/etc/os-release'


def _linux_sys_name() -> str:
    """
    Try to discover linux system name base on /etc/os-release file or lsb_release command output
    https://www.freedesktop.org/software/systemd/man/os-release.html
    """
    if os.path.exists(OS_RELEASE_PATH):
        with open(OS_RELEASE_PATH) as f_p:
            data = {}
            for line in f_p:
                field, value = line.split('=')
                data[field.strip()] = value.strip().strip('"')
        if 'PRETTY_NAME' in data:
            return data['PRETTY_NAME']
        if 'NAME' in data:
            if 'VERSION' in data:
                return f'{data["NAME"]} {data["VERSION"]}'
            if 'VERSION_ID' in data:
                return f'{data["NAME"]} {data["VERSION_ID"]}'
            return f'{data["NAME"]} (no version)'

    return _linux_sys_name_lsb_release()


def _linux_sys_name_lsb_release() -> str:
    """
    Try to discover linux system name base on lsb_release command output
    """
    with contextlib.suppress(subprocess.CalledProcessError):
        res = subprocess.run(
            ['lsb_release', '-d', '-r'], check=True, capture_output=True
        )
        text = res.stdout.decode()
        data = {}
        for line in text.split('\n'):
            key, val = line.split(':')
            data[key.strip()] = val.strip()
        version_str = data['Description']
        if not version_str.endswith(data['Release']):
            version_str += ' ' + data['Release']
        return version_str
    return ''


def _sys_name() -> str:
    """
    Discover MacOS or Linux Human readable information. For Linux provide information about distribution.
    """
    with contextlib.suppress(Exception):
        if sys.platform == 'linux':
            return _linux_sys_name()
        if sys.platform == 'darwin':
            with contextlib.suppress(subprocess.CalledProcessError):
                res = subprocess.run(
                    ['sw_vers', '-productVersion'],
                    check=True,
                    capture_output=True,
                )
                return f'MacOS {res.stdout.decode().strip()}'
    return ''


def _napari_from_conda() -> bool:
    """
    Try to check if napari was installed using conda.

    This is done by checking for the presence of a conda metadata json file
    in the current environment's conda-meta directory.

    Returns
    -------
    bool
        True if the main napari application is installed via conda, False otherwise.
    """
    # Check for napari-related conda metadata files
    napari_conda_files = list(
        Path(sys.prefix, 'conda-meta').glob('napari-*.json')
    )
    # Match only the napari package by using napari-<version>.json
    # This is to exclude plugins napari-svg, etc.
    napari_pattern = re.compile(r'^napari-\d+(\.\d+)*.*\.json$')

    return any(napari_pattern.match(file.name) for file in napari_conda_files)


def get_launch_command() -> str:
    """Get the information how the program was launched.

    Returns
    -------
    str
        The command used to launch the program.
    """

    return ' '.join(sys.argv)


def get_plugin_list() -> str:
    """Get a list of installed plugins.

    Returns
    -------
    str
        A string containing the names and versions of installed plugins.
    """
    try:
        from npe2 import PluginManager

        pm = PluginManager.instance()
        pm.discover(include_npe1=True)
        pm.index_npe1_adapters()  # type: ignore[no-untyped-call]
        fields = [
            'name',
            'package_metadata.version',
            'contributions',
        ]
        pm_dict = pm.dict(include=set(fields))

        res = []

        for plugin in pm_dict['plugins'].values():
            count_contributions = sum(
                len(x)
                for x in plugin['contributions'].values()
                if x is not None
            )
            res.append(
                f'  - {plugin["name"]}: {plugin["package_metadata"]["version"]} ({count_contributions} contributions)'
            )
        return '<br>'.join(res) + '<br>'
    except ImportError as e:  # pragma: no cover
        return f'Failed to load plugin information: <br> {e}'


def sys_info(as_html: bool = False) -> str:
    """Gathers relevant module versions for troubleshooting purposes.

    Parameters
    ----------
    as_html : bool
        if True, info will be returned as HTML, suitable for a QTextEdit widget
    """
    sys_version = sys.version.replace('\n', ' ')
    text = f'<b>napari</b>: {napari.__version__}'
    if _napari_from_conda():
        text += ' (from conda)'
    text += f'<br><b>Platform</b>: {platform.platform()}<br>'

    __sys_name = _sys_name()
    if __sys_name:
        text += f'<b>System</b>: {__sys_name}<br>'

    text += f'<b>Python</b>: {sys_version}<br>'

    try:
        from qtpy import API_NAME, PYQT_VERSION, PYSIDE_VERSION, QtCore

        if API_NAME in {'PySide2', 'PySide6'}:
            API_VERSION = PYSIDE_VERSION
        elif API_NAME in {'PyQt5', 'PyQt6'}:
            API_VERSION = PYQT_VERSION
        else:
            API_VERSION = ''

        text += (
            f'<b>Qt</b>: {QtCore.__version__}<br>'
            f'<b>{API_NAME}</b>: {API_VERSION}<br>'
        )

    except Exception as e:  # noqa BLE001
        text += f'<b>Qt</b>: Import failed ({e})<br>'

    modules = (
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('dask', 'Dask'),
        ('vispy', 'VisPy'),
        ('magicgui', 'magicgui'),
        ('superqt', 'superqt'),
        ('in_n_out', 'in-n-out'),
        ('app_model', 'app-model'),
        ('psygnal', 'psygnal'),
        ('npe2', 'npe2'),
        ('pydantic', 'pydantic'),
    )

    loaded = {}
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f'<b>{name}</b>: {version(module)}<br>'
        except PackageNotFoundError:
            text += f'<b>{name}</b>: Import failed<br>'

    text += '<br><b>OpenGL:</b><br>'

    try:
        from OpenGL.version import __version__ as pyopengl_version

        text += f'  - PyOpenGL: {pyopengl_version}<br>'
    except ImportError:
        text += '  - PyOpenGL: Import failed<br>'

    if loaded.get('vispy', False):
        from napari._vispy.utils.gl import get_max_texture_sizes

        sys_info_text = (
            '<br>'.join(
                [
                    loaded['vispy'].sys_info().split('\n')[index]
                    for index in [-4, -3]
                ]
            )
            .replace("'", '')
            .replace('<br>', '<br>  - ')
        )
        text += f'  - {sys_info_text}<br>'
        _, max_3d_texture_size = get_max_texture_sizes()
        text += f'  - GL_MAX_3D_TEXTURE_SIZE: {max_3d_texture_size}<br>'
    else:
        text += '  - failed to load vispy'

    text += '<br><b>Screens:</b><br>'

    try:
        from qtpy.QtGui import QGuiApplication

        screen_list = QGuiApplication.screens()
        for i, screen in enumerate(screen_list, start=1):
            text += f'  - screen {i}: resolution {screen.geometry().width()}x{screen.geometry().height()}, scale {screen.devicePixelRatio()}<br>'
    except Exception as e:  # noqa BLE001
        text += f'  - failed to load screen information {e}'

    text += '<br><b>Optional:</b><br>'

    optional_modules = (
        ('numba', 'numba'),
        ('triangle', 'triangle'),
        ('napari_plugin_manager', 'napari-plugin-manager'),
        ('bermuda', 'bermuda'),
        ('PartSegCore_compiled_backend', 'PartSegCore'),
    )

    for module, name in optional_modules:
        try:
            text += f'  - <b>{name}</b>: {version(module)}<br>'
        except PackageNotFoundError:
            text += f'  - {name} not installed<br>'

    try:
        from napari.settings import get_settings

        _async_setting = str(get_settings().experimental.async_)
        _autoswap_buffers = str(get_settings().experimental.autoswap_buffers)
        _triangulation_backend = str(
            get_settings().experimental.triangulation_backend
        )
        _config_path = get_settings().config_path
    except ValueError:
        from napari.utils._appdirs import user_config_dir

        _async_setting = str(os.getenv('NAPARI_ASYNC', 'False'))
        _autoswap_buffers = str(os.getenv('NAPARI_AUTOSWAP', 'False'))
        _triangulation_backend = str(
            os.getenv('NAPARI_TRIANGULATION_BACKEND', 'Fastest available')
        )
        _config_path = os.getenv('NAPARI_CONFIG', user_config_dir())

    text += '<br><b>Experimental Settings:</b><br>'
    text += f'  - Async: {_async_setting}<br>'
    text += f'  - Autoswap buffers: {_autoswap_buffers}<br>'
    text += f'  - Triangulation backend: {_triangulation_backend}<br>'

    text += '<br><b>Settings path:</b><br>'
    text += f'  - {_config_path}<br>'

    text += '<br><b>Launch command:</b><br>'
    text += f'  - {get_launch_command()}<br>'

    text += '<br><b>Plugins:</b><br>'
    text += get_plugin_list()

    if not as_html:
        text = (
            text.replace('<br>', '\n').replace('<b>', '').replace('</b>', '')
        )
    return text


citation_text = (
    'napari contributors (2019). napari: a '
    'multi-dimensional image viewer for python. '
    'doi:10.5281/zenodo.3555620'
)
