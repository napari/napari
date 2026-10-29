from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from napari.utils._platformdirs import user_cache_dir

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

LOADING_GIF_PATH = str((Path(__file__).parent / 'loading.gif').resolve())
ICON_PATH = (Path(__file__).parent / 'icons').resolve()
ICONS = {x.stem: str(x) for x in ICON_PATH.iterdir() if x.suffix == '.svg'}
PLUGIN_FILE_NAME = 'plugin.txt'
CACHE_FINGERPRINT_FILE = '_cache_fingerprint.txt'
_ICON_CACHE_VERSION = 1
THEME_OPACITIES = (0.5, 1)
THEME_OVERRIDES = {
    'warning': 'warning',
    'error': 'error',
    'logo_silhouette': 'foreground',
}


def get_icon_path(name: str) -> str:
    """Return path to an SVG in the theme icons."""
    if name not in ICONS:
        raise ValueError(
            f'unrecognized icon name: {name!r}. Known names: {set(ICONS)}'
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
    path_or_xml: str | Path, color: str | None = None, opacity: float = 1.0
) -> str:
    """Return a colorized version of the SVG XML at ``path``.

    Parameters
    ----------
    path_or_xml : str | Path
        Path to an SVG file, or raw SVG XML string.
    color : str, optional
        Color to use for the SVG, by default None. If None, the original SVG
        is returned.  If a string, it should be a valid SVG color style.
    opacity : float, optional
        Opacity to use for the SVG, by default 1.0.

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
        raise ValueError(f'Could not detect svg tag in {path_or_xml!r}')
    # use regex to find the svg tag and insert css right after
    # (the '\\1' syntax includes the matched tag in the output)
    return svg_elem.sub(f'\\1{svg_style.format(color, opacity)}', xml)


def generate_colorized_svgs(
    svg_paths: Iterable[str | Path],
    colors: Iterable[str | tuple[str, str]],
    opacities: Iterable[float] = (1.0,),
    theme_override: dict[str, str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Helper function to generate colorized SVGs.

    This is a generator that yields tuples of ``(alias, icon_xml)`` for every
    combination (Cartesian product) of `svg_path`, `color`, and `opacity`
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

    # mapping of svg_stem to theme_key
    theme_override = theme_override or {}

    ALIAS_T = '{color}/{svg_stem}{opacity}.svg'

    for color, path, op in product(colors, svg_paths, opacities):
        clrkey = color
        svg_stem = Path(path).stem
        if isinstance(color, tuple):
            from napari.utils.theme import get_theme

            clrkey, theme_key = color
            theme_key = theme_override.get(svg_stem, theme_key)
            color = getattr(get_theme(clrkey), theme_key).as_hex()
            # convert color to string to fit get_colorized_svg signature

        op_key = '' if op == 1 else f'_{op * 100:.0f}'
        alias = ALIAS_T.format(color=clrkey, svg_stem=svg_stem, opacity=op_key)
        yield alias, get_colorized_svg(path, color, op)


def write_colorized_svgs(
    dest: str | Path,
    svg_paths: Iterable[str | Path],
    colors: Iterable[str | tuple[str, str]],
    opacities: Iterable[float] = (1.0,),
    theme_override: dict[str, str] | None = None,
) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    svgs = generate_colorized_svgs(
        svg_paths=svg_paths,
        colors=colors,
        opacities=opacities,
        theme_override=theme_override,
    )

    for alias, svg in svgs:
        (dest / Path(alias).name).write_text(svg)


def _theme_path(theme_name: str) -> Path:
    return Path(user_cache_dir()) / '_themes' / theme_name


def _get_expected_svg_filenames(
    svg_stems: Iterable[str] | None = None,
    opacities: Iterable[float] = THEME_OPACITIES,
) -> list[str]:
    """Get the list of expected SVG filenames generated for a theme."""
    if svg_stems is None:
        svg_stems = ICONS.keys()
    filenames: list[str] = []
    for stem in sorted(svg_stems):
        for op in opacities:
            op_key = '' if op == 1 else f'_{op * 100:.0f}'
            filenames.append(f'{stem}{op_key}.svg')
    return filenames


def _compute_source_svgs_hash(
    icons: Mapping[str, str | Path] | None = None,
) -> str:
    """Compute a SHA-256 hash across all source icon SVG files directly from content."""
    icons_dict = ICONS if icons is None else dict(icons)
    hasher = hashlib.sha256()
    for stem, path_str in sorted(icons_dict.items()):
        path = Path(path_str)
        hasher.update(stem.encode('utf-8'))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _compute_theme_svg_fingerprint(
    theme_name: str,
    icons: Mapping[str, str | Path] | None = None,
) -> str:
    """Compute a deterministic hash covering all inputs for theme SVG generation."""
    from napari.utils.theme import get_theme

    theme = get_theme(theme_name)
    relevant_keys = sorted({'icon', *THEME_OVERRIDES.values()})
    color_values = {
        k: getattr(theme, k).as_hex()
        for k in relevant_keys
        if hasattr(theme, k)
    }

    fingerprint_data = {
        'version': _ICON_CACHE_VERSION,
        'style': svg_style,
        'opacities': list(THEME_OPACITIES),
        'theme_overrides': THEME_OVERRIDES,
        'colors': color_values,
        'source_svgs_hash': _compute_source_svgs_hash(icons),
    }
    payload = json.dumps(fingerprint_data, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _is_cache_valid(out: Path, expected_fingerprint: str) -> bool:
    """Check whether the theme SVG cache is complete and up to date."""
    fingerprint_path = out / CACHE_FINGERPRINT_FILE
    try:
        if not fingerprint_path.is_file():
            return False
        stored_fingerprint = fingerprint_path.read_text(
            encoding='utf-8'
        ).strip()
        if stored_fingerprint != expected_fingerprint:
            return False

        # Verify all expected output SVG files exist and are non-empty
        expected_files = _get_expected_svg_filenames()
        for filename in expected_files:
            file_path = out / filename
            if not file_path.is_file() or file_path.stat().st_size == 0:
                return False

        # Also verify plugin.txt exists, is readable, and is non-empty
        plugin_path = out / PLUGIN_FILE_NAME
        if not plugin_path.is_file():
            return False
        plugin_content = plugin_path.read_text(encoding='utf-8').strip()
        return bool(plugin_content)
    except (OSError, UnicodeDecodeError, ValueError):
        return False


def _remove_fingerprint_file(dest_dir: Path) -> None:
    """Remove fingerprint file so partial or failed writes are not validated.

    Raises OSError if an existing fingerprint file cannot be removed.
    """
    fingerprint_path = dest_dir / CACHE_FINGERPRINT_FILE
    fingerprint_path.unlink(missing_ok=True)


def _write_fingerprint_atomic(dest_dir: Path, fingerprint: str) -> None:
    """Write the fingerprint atomically to prevent partial cache validation."""
    dest_file = dest_dir / CACHE_FINGERPRINT_FILE
    temp_fd, temp_path = tempfile.mkstemp(
        dir=dest_dir,
        prefix=f'{CACHE_FINGERPRINT_FILE}.tmp',
    )
    closed = False
    try:
        with open(temp_fd, 'w', encoding='utf-8') as f:
            closed = True
            f.write(fingerprint)
        os.replace(temp_path, dest_file)
    except BaseException:
        if not closed:
            with suppress(OSError):
                os.close(temp_fd)
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass
        raise


def build_theme_svgs(theme_name: str, source: str) -> str:
    """Build theme-specific SVGs for a given theme and source.

    Parameters
    ----------
    theme_name : str
        The name of the theme.
    source : str
        Name of plugin providing theme.

    Returns
    -------
    str
        The path to the directory containing the generated SVGs.
    """
    out = _theme_path(theme_name)
    fingerprint = _compute_theme_svg_fingerprint(theme_name)

    if _is_cache_valid(out, fingerprint):
        plugin_file = out / PLUGIN_FILE_NAME
        try:
            current_source = plugin_file.read_text(encoding='utf-8')
        except OSError:
            current_source = None

        if current_source != source:
            plugin_file.write_text(source, encoding='utf-8')
        return str(out)

    # Invalidate existing fingerprint before writing new SVGs so an
    # interrupted write cannot validate partial output.
    _remove_fingerprint_file(out)

    try:
        # Invalidate in-memory SVG caches so rebuild uses fresh file contents
        get_raw_svg.cache_clear()
        get_colorized_svg.cache_clear()

        write_colorized_svgs(
            out,
            svg_paths=ICONS.values(),
            colors=[(theme_name, 'icon')],
            opacities=THEME_OPACITIES,
            theme_override=THEME_OVERRIDES,
        )
        with (out / PLUGIN_FILE_NAME).open('w', encoding='utf-8') as f:
            f.write(source)

        _write_fingerprint_atomic(out, fingerprint)
    except BaseException:
        with suppress(OSError):
            _remove_fingerprint_file(out)
        raise

    return str(out)
