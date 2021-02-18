from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generator, Iterable

QRC_TEMPLATE = """
<!DOCTYPE RCC><RCC version="1.0">
<qresource prefix="/themes">
{}
</qresource>
</RCC>
"""


DEFAULT_OPACITY = (0.5, 1)


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
    opacities = opacities or DEFAULT_OPACITY

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
    from subprocess import CalledProcessError, run

    import PySide2

    exe = Path(PySide2.__file__).parent / 'rcc'
    if not exe.exists():
        raise RuntimeError(f"PySide2 rcc binary not found at {exe}")

    cmd = [str(exe), '-g', 'python', '-o', outfile, qrc]
    try:
        run(cmd, check=True, capture_output=True)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to build PySide2 resources {e}")


def compile_qrc(qrc, outfile):
    from qtpy import API_NAME

    if API_NAME == 'PyQt5':
        _compile_qrc_pyqt5(qrc, outfile)
    elif API_NAME == 'PySide2':
        _compile_qrc_pyside2(qrc, outfile)
    else:
        raise RuntimeError(
            f"Cannot compile QRC. Unexpected qtpy API name: {API_NAME}"
        )


@lru_cache()
def build_qt_resources(themes=None, icons=None, opacities=None) -> str:
    with temporary_qrc_file(themes, icons, opacities) as qrc:
        with NamedTemporaryFile() as f:
            compile_qrc(qrc, f.name)
            f.seek(0)
            out = f.read().decode()
            return out.replace('PySide2', 'qtpy').replace('PyQt5', 'qtpy')


def register_resources(themes=None, icons=None, opacities=None):
    exec(build_qt_resources(themes, icons, opacities), globals())


if __name__ == '__main__':

    with open('/Users/talley/Desktop/out.py', 'w') as fn:
        fn.write(build_qt_resources())
