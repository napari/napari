import os.path
from functools import lru_cache
from glob import glob
from typing import List, Optional

from ._icons import _register_napari_resources, compile_qt_svgs
from ._svg import QColoredSVGIcon

__all__ = [
    'get_stylesheet',
    'QColoredSVGIcon',
    '_register_napari_resources',
    'compile_qt_svgs',
]


@lru_cache(maxsize=12)
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
    resources_dir = os.path.abspath(os.path.dirname(__file__))
    stylesheet = ''
    for file in sorted(glob(os.path.join(resources_dir, 'styles', '*.qss'))):
        with open(file) as f:
            stylesheet += f.read()
    if extra:
        for file in extra:
            with open(file) as f:
                stylesheet += f.read()

    if theme:
        from ...utils.theme import get_theme, template

        return template(stylesheet, **get_theme(theme))

    return stylesheet
