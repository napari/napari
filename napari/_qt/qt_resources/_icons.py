from __future__ import annotations

from contextlib import contextmanager
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generator, Iterable

import qtpy

QRC_TEMPLATE = """
<!DOCTYPE RCC><RCC version="1.0">
<qresource prefix="/themes">
{}
</qresource>
</RCC>
"""


DEFAULT_OPACITIES = (0.5, 1)


@contextmanager
def temporary_qrc_file(
    themes: Iterable[str] | None = None,
    icons: Iterable[str] | None = None,
    opacities: Iterable[float] | None = None,
) -> Generator[str, None, None]:
    """Create a qrc file for all combinations of ``icon``, ``theme``, ``opac``.

    Generates a .qrc file containing, for all theme/icons:

    <qresource prefix="/themes">
        <file alias='theme/icon_name.svg'>theme_icon_name.svg</file>
    </qresource>

    Parameters
    ----------
    themes : iterable of str, optional
        The theme names for which to generate icons, by default all available
        themes.
    icons : iterable of str, optional
        The set of icons , by default all available icons
    opacities : iterable of float, optional
        An iterable of icon opacities to pre-generate, by default (0.6, 1).
        Icons with opacity < 1 can be accessed with the opacity as a percentage
        for instance: ``icon_name_50.svg``

    Yields
    -------
    QRC_name : str
        The name of the temporary QRC file.  It will be in a temporary
        directory containing all of the necessary modified/colorized svgs.
    """
    from ...resources._icons import ICONS, get_colorized_svg
    from ...utils.theme import _themes

    themes = themes or _themes.keys()
    icons = icons or ICONS.keys()
    opacities = opacities or DEFAULT_OPACITIES

    # theme, icon_name, opacity_key, colorized.svg
    FILE_T = "<file alias='{}/{}{}.svg'>{}</file>"

    # mapping of icon_name to theme key, otherwise icon color will be "icon"
    color_override = {'warning': 'warning'}

    # create a temporary directory for the qrc file and all colorized svgs.
    with TemporaryDirectory() as tdir_name:
        tmp_dir = Path(tdir_name)

        # create the colorized SVGs
        files = []
        for theme, icon_name, op in product(_themes, ICONS, opacities):
            opacity_key = "" if op == 1 else f"_{op * 100:.0f}"
            tmp_svg = f"{theme}_{icon_name}{opacity_key}.svg"
            color = _themes[theme][color_override.get(icon_name, 'icon')]
            xml = get_colorized_svg(ICONS[icon_name], color, op)
            (tmp_dir / tmp_svg).write_text(xml)
            files.append(FILE_T.format(theme, icon_name, opacity_key, tmp_svg))

        # write the QRC file
        res_file = tmp_dir / 'res.qrc'
        res_file.write_text(QRC_TEMPLATE.format("\n".join(files)))
        yield str(res_file)


def _compile_qrc_pyqt5(qrc, outfile):
    from PyQt5.pyrcc_main import processResourceFile

    processResourceFile([qrc], outfile, False)


def _compile_qrc_pyside2(qrc, outfile):
    import os
    from subprocess import CalledProcessError, run

    import PySide2

    pyside_root = Path(PySide2.__file__).parent

    if os.name == 'nt':
        look_for = ('rcc.exe', 'pyside2-rcc.exe')
    else:
        look_for = ('rcc', 'pyside2-rcc')

    for bin in look_for:
        if (pyside_root / bin).exists():
            cmd = [str(pyside_root / bin)]
            if 'pyside2' not in bin:  # older version
                cmd.extend(['-g', 'python'])
            break
    else:
        raise RuntimeError(f"PySide2 rcc binary not found in {pyside_root}")

    try:
        run(cmd + ['-o', outfile, qrc], check=True, capture_output=True)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to build PySide2 resources {e}")


def compile_qrc(qrc, outfile):

    if qtpy.API_NAME == 'PyQt5':
        _compile_qrc_pyqt5(qrc, outfile)
    elif qtpy.API_NAME == 'PySide2':
        _compile_qrc_pyside2(qrc, outfile)
    else:
        raise RuntimeError(
            f"Cannot compile QRC. Unexpected qtpy API name: {qtpy.API_NAME}"
        )


def build_qt_resources(themes=None, icons=None, opacities=None) -> str:
    with temporary_qrc_file(themes, icons, opacities) as qrc:
        with NamedTemporaryFile() as f:
            compile_qrc(qrc, f.name)
            f.seek(0)
            out = f.read().decode()
            return out.replace('PySide2', 'qtpy').replace('PyQt5', 'qtpy')


def register_qt_resources(themes=None, icons=None, opacities=None):
    exec(build_qt_resources(themes, icons, opacities), globals())


def register_napari_resources(persist=True, force_rebuild=False):
    """Build, save, and register napari Qt icon resources.

    If a _qt_resources*.py file exists that matches the qt version and hash of
    the icons directory, this function will simply import that file.
    Otherwise, this function will create themed versions of the icons in the
    resources/icons folder, create a .qrc file, and compile it, and import it.
    If `persist` is `True`, the compiled resources will be saved for next run.

    Parameters
    ----------
    persist : bool, optional
        Whether to save newly compiled resources, by default True
    force_rebuild : bool, optional
        If true, resources will be recompiled and resaved, even if they already
        exist
    """
    from ...resources._icons import ICON_PATH
    from ...utils.misc import dir_hash

    icon_hash = dir_hash(ICON_PATH)  # get hash of icons folder contents
    key = f'_qt_resources_{qtpy.API_NAME}_{qtpy.QT_VERSION}_{icon_hash}'
    key = key.replace(".", "_")
    save_path = Path(__file__).parent / f"{key}.py"

    if not force_rebuild and save_path.exists():
        from importlib import import_module

        modname = f'napari._qt.qt_resources.{key}'
        import_module(modname)
    else:
        resource_file = build_qt_resources()
        if persist:
            try:
                save_path.write_text(resource_file)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to save qt_resources to {save_path}: {e}"
                )
        exec(resource_file, globals())
