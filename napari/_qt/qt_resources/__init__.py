import warnings
from pathlib import Path
from typing import List, Optional

from ._icons import (
    _register_napari_resources,
    _unregister_napari_resources,
    compile_qt_svgs,
    register_napari_themes,
)
from ._svg import QColoredSVGIcon

__all__ = [
    'get_stylesheet',
    'QColoredSVGIcon',
    '_register_napari_resources',
    '_unregister_napari_resources',
    'compile_qt_svgs',
    'register_napari_themes',
]

STYLE_PATH = (Path(__file__).parent / 'styles').resolve()
STYLES = {x.stem: str(x) for x in STYLE_PATH.iterdir() if x.suffix == '.qss'}


def get_stylesheet(
    theme: str = None, extra: Optional[List[str]] = None
) -> str:
    """Combine all qss files into single, possibly pre-themed, style string.

    Parameters
    ----------
    theme : str, optional
        Theme to apply to the stylesheet. If no theme is provided, the returned
        stylesheet will still have ``{{ template_variables }}`` that need to be
        replaced using the :func:`napari.utils.theme.template` function prior
        to using the stylesheet.
    extra : list of str, optional
        Additional paths to QSS files to include in stylesheet, by default None

    Returns
    -------
    css : str
        The combined stylesheet.
    """
    stylesheet = ''
    for key in sorted(STYLES.keys()):
        file = STYLES[key]
        with open(file) as f:
            stylesheet += f.read()
    if extra:
        for file in extra:
            with open(file) as f:
                stylesheet += f.read()

    if theme:
        from ...utils.theme import get_theme, template

        return template(stylesheet, **get_theme(theme, as_dict=True))

    return stylesheet
