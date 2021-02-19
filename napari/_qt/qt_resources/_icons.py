import os
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

import qtpy

QRC_TEMPLATE = """
<!DOCTYPE RCC><RCC version="1.0">
<qresource prefix="{}">
{}
</qresource>
</RCC>
"""
ALIAS_T = '{color}/{svg_stem}{opacity}.svg'
FILE_T = "<file alias='{alias}'>{path}</file>"

DEFAULT_OPACITIES = (0.5, 1)


@contextmanager
def temporary_qrc_file(
    xmls: Iterable[Tuple[str, str]], prefix: str = ''
) -> Iterator[str]:
    # create a temporary directory for the qrc file and all colorized svgs.
    with TemporaryDirectory() as tdir_name:
        tmp_dir = Path(tdir_name)

        # create the colorized SVGs
        files = []
        for alias, xml in xmls:
            path = alias.replace("/", "_")
            (tmp_dir / path).write_text(xml)
            files.append(FILE_T.format(alias=alias, path=path))

        # write the QRC file
        res_file = tmp_dir / 'res.qrc'
        res_file.write_text(QRC_TEMPLATE.format(prefix, '\n'.join(files)))
        yield str(res_file)


def generate_colorized_svgs(
    svg_paths: Iterable[Union[str, Path]],
    colors: Iterable[Union[str, Tuple[str, str]]],
    opacities: Iterable[float] = (1.0,),
    theme_override: Optional[Dict[str, str]] = None,
) -> Iterator[Tuple[str, str]]:
    from ...resources._icons import get_colorized_svg

    # mapping of svg_stem to theme_key
    theme_override = theme_override or dict()

    for color, path, op in product(colors, svg_paths, opacities):
        clrkey = color
        svg_stem = Path(path).stem
        if isinstance(color, tuple):
            from ...utils.theme import get_theme

            clrkey, theme_key = color
            theme_key = theme_override.get(theme_key, theme_key)
            color = get_theme(clrkey)[theme_key]

        op_key = "" if op == 1 else f"_{op * 100:.0f}"
        alias = ALIAS_T.format(color=clrkey, svg_stem=svg_stem, opacity=op_key)
        yield (alias, get_colorized_svg(path, color, op))


def _compile_qrc_pyqt5(qrc) -> bytes:
    from PyQt5.pyrcc_main import processResourceFile

    tf = NamedTemporaryFile(suffix='.py', delete=False)
    try:
        tf.close()
        processResourceFile([qrc], tf.name, False)
        out = Path(tf.name).read_bytes()
    finally:
        os.unlink(tf.name)
    return out


def _compile_qrc_pyside2(qrc) -> bytes:
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
        return run(cmd + [qrc], check=True, capture_output=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to build PySide2 resources {e}")


def compile_qrc(qrc) -> bytes:
    if qtpy.API_NAME == 'PyQt5':
        return _compile_qrc_pyqt5(qrc).replace(b'PyQt5', b'qtpy')
    elif qtpy.API_NAME == 'PySide2':
        return _compile_qrc_pyside2(qrc).replace(b'PySide2', b'qtpy')
    else:
        raise RuntimeError(
            f"Cannot compile QRC. Unexpected qtpy API name: {qtpy.API_NAME}"
        )


def compile_qt_resources(
    svg_paths: Iterable[Union[str, Path]],
    colors: Iterable[Union[str, Tuple[str, str]]],
    opacities: Iterable[float] = (1.0,),
    theme_override: Optional[Dict[str, str]] = None,
    prefix='',
    save_path: Optional[str] = None,
    register: bool = False,
) -> str:

    svgs = generate_colorized_svgs(
        svg_paths, colors, opacities, theme_override
    )

    with temporary_qrc_file(svgs, prefix=prefix) as qrc:
        output = compile_qrc(qrc)
        if save_path:
            Path(save_path).write_bytes(output)
        if register:
            from ..qt_event_loop import get_app

            get_app()  # make sure app is created before we do this
            exec(output, globals())
        return output.decode()


def _compile_napari_resources(
    save_path: Optional[Union[str, Path]] = None
) -> str:
    from ...resources._icons import ICONS
    from ...utils.theme import _themes

    svgs = generate_colorized_svgs(
        svg_paths=ICONS.values(),
        colors=[(k, 'icon') for k in _themes],
        opacities=(0.5, 1),
        theme_override={'warning': 'warning'},
    )

    with temporary_qrc_file(svgs, prefix='themes') as qrc:
        output = compile_qrc(qrc)
        if save_path:
            Path(save_path).write_bytes(output)
        return output.decode()


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

    force = force_rebuild or os.environ.get('NAPARI_REBUILD_RESOURCES')

    if not force and save_path.exists():
        from importlib import import_module

        modname = f'napari._qt.qt_resources.{key}'
        import_module(modname)
    else:
        resources = _compile_napari_resources(save_path=persist and save_path)
        exec(resources, globals())
