# syntax_style for the console must be one of the supported styles from
# pygments - see here for examples https://help.farbox.com/pygments.html
import re
import warnings
from ast import literal_eval
from typing import Union

from pydantic import validator
from pydantic.color import Color

from .._vendor import darkdetect
from ..resources._icons import build_theme_svgs
from ..utils.translations import trans
from .events import EventedModel
from .events.containers._evented_dict import EventedDict

try:
    from qtpy import QT_VERSION

    major, minor, *rest = QT_VERSION.split('.')
    use_gradients = (int(major) >= 5) and (int(minor) >= 12)
except Exception:
    use_gradients = False


class Theme(EventedModel):
    """Theme model.

    Attributes
    ----------
    name : str
        Name of the virtual folder where icons will be saved to.
    syntax_style : str
        Name of the console style.
        See for more details: https://pygments.org/docs/styles/
    canvas : Color
        Background color of the canvas.
    background : Color
        Color of the application background.
    foreground : Color
        Color to contrast with the background.
    primary : Color
        Color used to make part of a widget more visible.
    secondary : Color
        Alternative color used to make part of a widget more visible.
    highlight : Color
        Color used to highlight visual element.
    text : Color
        Color used to display text.
    warning : Color
        Color used to indicate something is wrong.
    current : Color
        Color used to highlight Qt widget.
    """

    name: str
    syntax_style: str
    canvas: Color
    console: Color
    background: Color
    foreground: Color
    primary: Color
    secondary: Color
    highlight: Color
    text: Color
    icon: Color
    warning: Color
    current: Color

    @validator("syntax_style", pre=True)
    def _ensure_syntax_style(value: str) -> str:
        from pygments.styles import STYLE_MAP

        assert value in STYLE_MAP, trans._(
            "Incorrect `syntax_style` value provided. Please use one of the following: {syntax_style}",
            deferred=True,
            syntax_style=f" {', '.join(STYLE_MAP)}",
        )
        return value


gradient_pattern = re.compile(r'([vh])gradient\((.+)\)')
darken_pattern = re.compile(r'{{\s?darken\((\w+),?\s?([-\d]+)?\)\s?}}')
lighten_pattern = re.compile(r'{{\s?lighten\((\w+),?\s?([-\d]+)?\)\s?}}')
opacity_pattern = re.compile(r'{{\s?opacity\((\w+),?\s?([-\d]+)?\)\s?}}')


def darken(color: Union[str, Color], percentage=10):
    if isinstance(color, str) and color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    else:
        color = color.as_rgb_tuple()
    ratio = 1 - float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red * ratio), 0), 255)
    green = min(max(int(green * ratio), 0), 255)
    blue = min(max(int(blue * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def lighten(color: Union[str, Color], percentage=10):
    if isinstance(color, str) and color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    else:
        color = color.as_rgb_tuple()
    ratio = float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red + (255 - red) * ratio), 0), 255)
    green = min(max(int(green + (255 - green) * ratio), 0), 255)
    blue = min(max(int(blue + (255 - blue) * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def opacity(color: Union[str, Color], value=255):
    if isinstance(color, str) and color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    else:
        color = color.as_rgb_tuple()
    red, green, blue = color
    return f'rgba({red}, {green}, {blue}, {max(min(int(value), 255), 0)})'


def gradient(stops, horizontal=True):
    if not use_gradients:
        return stops[-1]

    if horizontal:
        grad = 'qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, '
    else:
        grad = 'qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, '

    _stops = [f'stop: {n} {stop}' for n, stop in enumerate(stops)]
    grad += ", ".join(_stops) + ")"

    return grad


def template(css: str, **theme):
    def darken_match(matchobj):
        color, percentage = matchobj.groups()
        return darken(theme[color], percentage)

    def lighten_match(matchobj):
        color, percentage = matchobj.groups()
        return lighten(theme[color], percentage)

    def opacity_match(matchobj):
        color, percentage = matchobj.groups()
        return opacity(theme[color], percentage)

    def gradient_match(matchobj):
        horizontal = matchobj.groups()[1] == 'h'
        stops = [i.strip() for i in matchobj.groups()[1].split('-')]
        return gradient(stops, horizontal)

    for k, v in theme.items():
        css = gradient_pattern.sub(gradient_match, css)
        css = darken_pattern.sub(darken_match, css)
        css = lighten_pattern.sub(lighten_match, css)
        css = opacity_pattern.sub(opacity_match, css)
        if isinstance(v, Color):
            v = v.as_rgb()
        css = css.replace('{{ %s }}' % k, v)
    return css


def get_system_theme() -> str:
    """Return the system default theme, either 'dark', or 'light'."""
    try:
        name = darkdetect.theme().lower()
    except Exception:
        name = "dark"

    return name


def get_theme(name, as_dict=None):
    """Get a copy of theme based on it's name.

    If you get a copy of the theme, changes to the theme model will not be
    reflected in the UI unless you replace or add the modified theme to
    the `_themes` container.

    Parameters
    ----------
    name : str
        Name of requested theme.
    as_dict : bool
        Flag to indicate that the old-style dictionary
        should be returned. This will emit deprecation warning.

    Returns
    -------
    theme: dict of str: str
        Theme mapping elements to colors. A copy is created
        so that manipulating this theme can be done without
        side effects.
    """
    if name == "system":
        name = get_system_theme()

    if name not in _themes:
        raise ValueError(
            trans._(
                "Unrecognized theme {name}. Available themes are {themes}",
                deferred=True,
                name=name,
                themes=available_themes(),
            )
        )
    theme = _themes[name]
    _theme = theme.copy()
    if as_dict is None:
        warnings.warn(
            trans._(
                "Themes were changed to use evented model with Pydantic's color type rather than the `rgb(x, y, z)`. The `as_dict=True` option will be changed to `as_dict=False` in 0.4.15",
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        as_dict = True
    if as_dict:
        _theme = _theme.dict()
        _theme = {
            k: v if not isinstance(v, Color) else v.as_rgb()
            for (k, v) in _theme.items()
        }
        return _theme
    return _theme


_themes: EventedDict[str, Theme] = EventedDict(basetype=Theme)


def register_theme(name, theme):
    """Register a new or updated theme.

    Parameters
    ----------
    name : str
        Name of requested theme.
    theme : dict of str: str, Theme
        Theme mapping elements to colors.
    """
    if isinstance(theme, dict):
        theme = Theme(**theme)
    assert isinstance(theme, Theme)
    _themes[name] = theme

    build_theme_svgs(name)


def unregister_theme(name):
    """Remove existing theme.

    Parameters
    ----------
    name : str
        Name of the theme to be removed.
    """
    _themes.pop(name, None)


def available_themes():
    """List available themes.

    Returns
    -------
    list of str
        Names of available themes.
    """
    return tuple(_themes) + ("system",)


def rebuild_theme_settings():
    """update theme information in settings.

    here we simply update the settings to reflect current list of available
    themes.
    """
    from ..settings import get_settings

    settings = get_settings()
    settings.appearance.refresh_themes()


DARK = Theme(
    name='dark',
    background='rgb(38, 41, 48)',
    foreground='rgb(65, 72, 81)',
    primary='rgb(90, 98, 108)',
    secondary='rgb(134, 142, 147)',
    highlight='rgb(106, 115, 128)',
    text='rgb(240, 241, 242)',
    icon='rgb(209, 210, 212)',
    warning='rgb(153, 18, 31)',
    current='rgb(0, 122, 204)',
    syntax_style='native',
    console='rgb(18, 18, 18)',
    canvas='black',
)
LIGHT = Theme(
    name='light',
    background='rgb(239, 235, 233)',
    foreground='rgb(214, 208, 206)',
    primary='rgb(188, 184, 181)',
    secondary='rgb(150, 146, 144)',
    highlight='rgb(163, 158, 156)',
    text='rgb(59, 58, 57)',
    icon='rgb(107, 105, 103)',
    warning='rgb(255, 18, 31)',
    current='rgb(253, 240, 148)',
    syntax_style='default',
    console='rgb(255, 255, 255)',
    canvas='white',
)

register_theme('dark', DARK)
register_theme('light', LIGHT)


# this function here instead of plugins._npe2 to avoid circular import
def _install_npe2_themes(_themes):
    import npe2

    for theme in npe2.PluginManager.instance().iter_themes():
        # `theme.type` is dark/light and supplies defaults for keys that
        # are not provided by the plugin
        d = _themes[theme.type].dict()
        d.update(theme.colors.dict(exclude_unset=True))
        register_theme(theme.id, d)


_install_npe2_themes(_themes)
_themes.events.added.connect(rebuild_theme_settings)
_themes.events.removed.connect(rebuild_theme_settings)
