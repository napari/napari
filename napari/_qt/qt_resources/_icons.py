"""Helper functions to turn SVGs into compiled Qt resources.

The primary *internal* function is :func:`_register_napari_resources`, which:

1.  Checks to see if there is a pre-existing & current `_qt_resources*.py` file
    in ``napari/_qt/qt_resources``.  The file name contains the Qt backend that
    was used to compile the resources, and a hash of the content of the icons
    folder.  If either are stale, a new file is generated following the steps
    below (unless the `NAPARI_REBUILD_RESOURCES` environment flag is set),
    otherwise the pre-existing file is imported.
2.  Colorizes all of the icons in the resources/icons folder, using all of the
    colors in the available themes, by adding a ``<style>`` element to the raw
    SVG XML. (see :func:`generate_colorized_svgs`)
3.  Builds a temporary .qrc file pointing to the colored icons, following the
    format described at https://doc.qt.io/qt-5/resources.html (see
    :func:`_temporary_qrc_file`)
4.  Compiles that .qrc file using the `resource compiler (rcc)
    <https://doc.qt.io/qt-5/rcc.html>`_ from the currently active Qt backend.
    The output is raw ``bytes``. (see :func:`compile_qrc`)
5.  The ``bytes`` from :func:`compile_qrc` are saved for later reloading (see
    first step), and are immediately executed with ``exec``, which has the
    effect of registering the resources.

The primary *external* function is :func:`compile_qt_svgs`.  It provides a
convenience wrapper around :func:`generate_colorized_svgs`,
:func:`_temporary_qrc_file`, and :func:`compile_qrc`, and performs a sequence of
steps to similar those described above, using arbitrary SVGs and colors as
inputs.
"""

import os
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import qtpy

from ...utils.translations import trans

__all__ = [
    '_register_napari_resources',
    'compile_qt_svgs',
    'compile_qrc',
    'generate_colorized_svgs',
]

QRC_TEMPLATE = """
<!DOCTYPE RCC><RCC version="1.0">
<qresource prefix="{}">
{}
</qresource>
</RCC>
"""
ALIAS_T = '{color}/{svg_stem}{opacity}.svg'
FILE_T = "<file alias='{alias}'>{path}</file>"

# This variable is updated by either `_register_napari_resources`
# and turned into a function which removes existing resources from the app.
_clear_resources: Optional[Callable] = None


@contextmanager
def _temporary_qrc_file(
    xmls: Iterable[Tuple[str, str]], prefix: str = ''
) -> Iterator[str]:
    """Create a temporary directory with icons and .qrc file

    This constructs a temporary directory and builds a .qrc file as described
    in https://doc.qt.io/qt-5/resources.html

    Parameters
    ----------
    xmls : Iterable[Tuple[str, str]]
        An iterable of ``(alias, xml)`` pairs, where `alias` is the name that
        you want to use to access the icon in the Qt Resource system (such as
        QSS), and `xml` is the *raw* SVG text (as read from a file, perhaps
        pre-colored using one of the below functions).
        The output of :func:`generate_colorized_svgs` is a suitable input for
        `xmls`.
    prefix : str, optional
        A prefix to use when accessing these resources.  For instance, if
        ``prefix='theme'``, and one of the aliases in `xmls` is
        `"my_icon.svg"`, then you could access the compiled resources at
        `:/theme/my_icon.svg`. by default ''

    Yields
    ------
    Iterator[str]
        Yields the *name* of the temporary generated .qrc file, which can be
        used as input to :func:`compile_qrc`.
    """
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
    """Helper function to generate colorized SVGs.

    This is a generator that yields tuples of ``(alias, icon_xml)`` for every
    combination (cartesian product) of `svg_path`, `color`, and `opacity`
    provided. It can be used as input to :func:`_temporary_qrc_file`.

    Parameters
    ----------
    svg_paths : Iterable[Union[str, Path]]
        An iterable of paths to svg files
    colors : Iterable[Union[str, Tuple[str, str]]]
        An iterable of colors.  Every icon will be generated in every color. If
        a `color` item is a string, it should be valid svg color style. Items
        may also be a 2-tuple of strings, in which case the first item should
        be an available theme name
        (:func:`~napari.utils.theme.available_themes`), and the second item
        should be a key in the theme (:func:`~napari.utils.theme.get_theme`),
    opacities : Iterable[float], optional
        An iterable of opacities to generate, by default (1.0,) Opacities less
        than one can be accessed in qss with the opacity as a percentage
        suffix, e.g.: ``my_svg_50.svg`` for opacity 0.5.
    theme_override : Optional[Dict[str, str]], optional
        When one of the `colors` is a theme ``(name, key)`` tuple,
        `theme_override` may be used to override the `key` for a specific icon
        name in `svg_paths`.  For example ``{'exclamation': 'warning'}``, would
        use the theme "warning" color for any icon named "exclamation.svg" by
        default `None`

    Yields
    ------
    (alias, xml) : Iterator[Tuple[str, str]]
        `alias` is the name that will used to access the icon in the Qt
        Resource system (such as QSS), and `xml` is the *raw* colorzied SVG
        text (as read from a file, perhaps pre-colored using one of the below
        functions).
    """
    from ...resources._icons import get_colorized_svg

    # mapping of svg_stem to theme_key
    theme_override = theme_override or dict()

    for color, path, op in product(colors, svg_paths, opacities):
        clrkey = color
        svg_stem = Path(path).stem
        if isinstance(color, tuple):
            from ...utils.theme import get_theme

            clrkey, theme_key = color
            theme_key = theme_override.get(svg_stem, theme_key)
            color = getattr(get_theme(clrkey, False), theme_key)

        op_key = "" if op == 1 else f"_{op * 100:.0f}"
        alias = ALIAS_T.format(color=clrkey, svg_stem=svg_stem, opacity=op_key)
        yield (alias, get_colorized_svg(path, color, op))


def _compile_qrc_pyqt5(qrc) -> bytes:
    """Compile qrc file using the PyQt5 method.

    PyQt5 compiles qrc files using a direct function from the shared library
    PyQt5.pyrcc.  They provide access via a helper function in `pyrcc_main`.
    """
    from PyQt5.pyrcc_main import processResourceFile

    # could not capture stdout no matter what I tried, so using a temp file.
    # need to do it this way instead of context manager because of windows'
    # handling of "shared" temporary files.
    tf = NamedTemporaryFile(suffix='.py', delete=False)
    try:
        tf.close()
        processResourceFile([qrc], tf.name, False)
        out = Path(tf.name).read_bytes()
    finally:
        os.unlink(tf.name)
    return out


def _compile_qrc_pyqt6(qrc) -> bytes:
    """
    We can't do that with PyQt6,

    The maintainer has discontinued pyrcc for PyQt6

    """

    raise NotImplementedError('pyrcc discontinued on Pyqt6')


def _compile_qrc_pyside2(qrc) -> bytes:
    """Compile qrc file using the PySide2 method.

    PySide compiles qrc files using the rcc binary in the root directory, (same
    path as PySide2.__init__)
    """
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
            if 'pyside2' not in bin:
                # the newer pure rcc version requires this for python
                cmd.extend(['-g', 'python'])
            break
    else:
        raise RuntimeError(f"PySide2 rcc binary not found in {pyside_root}")

    try:
        return run(cmd + [qrc], check=True, capture_output=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to build PySide2 resources {e}")


def compile_qrc(qrc) -> bytes:
    """Compile a qrc file into a resources.py bytes"""
    if qtpy.API_NAME == 'PyQt6':
        return _compile_qrc_pyqt6(qrc).replace(b'PyQt6', b'qtpy')
    elif qtpy.API_NAME == 'PyQt5':
        return _compile_qrc_pyqt5(qrc).replace(b'PyQt5', b'qtpy')
    elif qtpy.API_NAME == 'PySide2':
        return _compile_qrc_pyside2(qrc).replace(b'PySide2', b'qtpy')
    else:
        raise RuntimeError(
            f"Cannot compile QRC. Unexpected qtpy API name: {qtpy.API_NAME}"
        )


def compile_qt_svgs(
    svg_paths: Iterable[Union[str, Path]],
    colors: Iterable[Union[str, Tuple[str, str]]],
    opacities: Iterable[float] = (1.0,),
    theme_override: Optional[Dict[str, str]] = None,
    prefix='',
    save_path: Optional[str] = None,
    register: bool = False,
) -> str:
    """Return compiled qt resources for all combinations of `svgs` and `colors`.

    This function is a convenience wrapper around
    :func:`generate_colorized_svgs`, :func:`_temporary_qrc_file`, and
    :func:`compile_qrc`.  It generates styled XML from SVG paths, organizes
    a `.qrc file <https://doc.qt.io/qt-5/resources.html>`_, compiles that file,
    and returns the text of the compiled python file (optionally saving it to
    `save_path` and/or immediately registering/importing it with `register`.)

    Parameters
    ----------
    svg_paths : Iterable[Union[str, Path]]
        An iterable of paths to svg files
    colors : Iterable[Union[str, Tuple[str, str]]]
        An iterable of colors.  Every icon will be generated in every color. If
        a `color` item is a string, it should be valid svg color style. Items
        may also be a 2-tuple of strings, in which case the first item should
        be an available theme name
        (:func:`~napari.utils.theme.available_themes`), and the second item
        should be a key in the theme (:func:`~napari.utils.theme.get_theme`),
    opacities : Iterable[float], optional
        An iterable of opacities to generate, by default (1.0,) Opacities less
        than one can be accessed in qss with the opacity as a percentage
        suffix, e.g.: ``my_svg_50.svg`` for opacity 0.5.
    theme_override : dict, optional
        When one of the `colors` is a theme ``(name, key)`` tuple,
        `theme_override` may be used to override the `key` for a specific icon
        name in `svg_paths`.  For example `{'exclamation': 'warning'}`, would
        use the theme "warning" color for any icon named "exclamation.svg" by
        default None
    prefix : str, optional
        A prefix to use when accessing these resources.  For instance, if
        ``prefix='theme'``, and one of the aliases in `xmls` is
        `"my_icon.svg"`, then you could access the compiled resources at
        `:/theme/my_icon.svg`. by default ''
    save_path : str, optional
        An optional output path for the compiled py resources, by default None
    register : bool, optional
        If `True`, immediately execute (import) the compiled resources.
        By default `False`

    Returns
    -------
    resources : str
        compiled resources as resources.py file
    """

    svgs = generate_colorized_svgs(
        svg_paths, colors, opacities, theme_override
    )

    with _temporary_qrc_file(svgs, prefix=prefix) as qrc:
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
    """Internal function to compile all napari icons for all themes."""
    from ...resources._icons import ICONS
    from ...utils.theme import _themes

    svgs = generate_colorized_svgs(
        svg_paths=ICONS.values(),
        colors=[(k, 'icon') for k in _themes],
        opacities=(0.5, 1),
        theme_override={'warning': 'warning', 'logo_silhouette': 'background'},
    )

    with _temporary_qrc_file(svgs, prefix='themes') as qrc:
        output = compile_qrc(qrc)
        if save_path:
            try:
                Path(save_path).write_bytes(output)
            except OSError as e:
                import warnings

                msg = trans._(
                    "Unable to save qt-resources: {err}",
                    err=str(e),
                    deferred=True,
                )
                warnings.warn(msg)

        return output.decode()


def _get_resources_path() -> Tuple[str, Path]:
    from ...resources._icons import ICONS
    from ...utils.misc import paths_hash

    icon_hash = paths_hash(ICONS.values())
    key = f'_qt_resources_{qtpy.API_NAME}_{qtpy.QT_VERSION}_{icon_hash}'
    key = key.replace(".", "_")
    save_path = Path(__file__).parent / f"{key}.py"
    return key, save_path


def _register_napari_resources(persist=True, force_rebuild=False):
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
        exist.  Rebuild may also be forced using the environment variable
        "NAPARI_REBUILD_RESOURCES".
    """
    # I hate to do this, but if we want to be able to dynamically change color
    # of e.g. icons, we must unregister old resources using the `qCleanupResources`
    # method which can be found in the `_qt_resources*.py` module.
    # If the resources are not cleared, it's 100% going to segfault which
    # is not desired. See: https://github.com/napari/napari/pull/2900
    global _clear_resources

    key, save_path = _get_resources_path()
    force = force_rebuild or os.environ.get('NAPARI_REBUILD_RESOURCES')

    if not force and save_path.exists():
        from importlib import import_module

        modname = f'napari._qt.qt_resources.{key}'
        mod = import_module(modname)
        _clear_resources = getattr(mod, "qCleanupResources")
    else:
        resources = _compile_napari_resources(save_path=persist and save_path)
        exec(resources, globals())
        _clear_resources = globals()["qCleanupResources"]


def _unregister_napari_resources():
    """Unregister resources."""
    global _clear_resources
    if _clear_resources is not None:
        _clear_resources()
        # since we've just unregistered resources, reference to this method should be
        # removed
        _clear_resources = None
    else:
        from importlib import import_module

        key, save_path = _get_resources_path()
        if save_path.exists():
            modname = f'napari._qt.qt_resources.{key}'
            mod = import_module(modname)
            qCleanupResources = getattr(mod, "qCleanupResources")
            qCleanupResources()  # try cleaning up resources


def register_napari_themes(event=None):
    """Register theme.

    This function takes care of unregistering existing resources and
    registering new or updated resources. This is necessary in order to
    add new icon(s) or update icon color.

    Not unregistering resources can lead to segfaults which is not desirable...
    """
    _unregister_napari_resources()
    _register_napari_resources(False, force_rebuild=True)
