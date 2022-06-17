import re
from functools import lru_cache
from pathlib import Path
from typing import Union

from ..utils.translations import trans

ICON_PATH = (Path(__file__).parent / 'icons').resolve()
ICONS = {x.stem: str(x) for x in ICON_PATH.iterdir() if x.suffix == '.svg'}


def get_icon_path(name: str) -> str:
    """Return path to an SVG in the theme icons."""
    if name not in ICONS:
        raise ValueError(
            trans._(
                "unrecognized icon name: {name!r}. Known names: {icons}",
                deferred=True,
                name=name,
                icons=set(ICONS),
            )
        )
    return ICONS[name]


svg_elem = re.compile(r'(<svg[^>]*>)')
svg_style = """<style type="text/css">
path {{fill: {0}; opacity: {1};}}
polygon {{fill: {0}; opacity: {1};}}
circle {{fill: {0}; opacity: {1};}}
rect {{fill: {0}; opacity: {1};}}
</style>"""


@lru_cache
def get_raw_svg(path: str) -> str:
    """Get and cache SVG XML."""
    return Path(path).read_text()


@lru_cache
def get_colorized_svg(
    path_or_xml: Union[str, Path], color: str = None, opacity=1
) -> str:
    """Return a colorized version of the SVG XML at ``path``.

    Raises
    ------
    ValueError
        If the path exists but does not contain valid SVG data.
    """
    path_or_xml = str(path_or_xml)
    xml = path_or_xml if '</svg>' in path_or_xml else get_raw_svg(path_or_xml)
    if not color:
        return xml

    if not svg_elem.search(xml):
        raise ValueError(
            trans._(
                "Could not detect svg tag in {path_or_xml!r}",
                deferred=True,
                path_or_xml=path_or_xml,
            )
        )
    # use regex to find the svg tag and insert css right after
    # (the '\\1' syntax includes the matched tag in the output)
    return svg_elem.sub(f'\\1{svg_style.format(color, opacity)}', xml)
