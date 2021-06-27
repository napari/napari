# syntax_style for the console must be one of the supported styles from
# pygments - see here for examples https://help.farbox.com/pygments.html
import re
import warnings
from ast import literal_eval

try:
    from qtpy import QT_VERSION

    major, minor, *rest = QT_VERSION.split('.')
    use_gradients = (int(major) >= 5) and (int(minor) >= 12)
except Exception:
    use_gradients = False

from ..utils.translations import trans


def __getattr__(attr):
    if attr == "palettes":
        warnings.warn(
            trans._(
                "palette is deprecated and will be removed after version 0.4.6. Please use get_theme and register_theme instead",
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return _themes
    raise AttributeError


_themes = {
    'dark': {
        'folder': 'dark',
        'background': 'rgb(38, 41, 48)',
        'foreground': 'rgb(65, 72, 81)',
        'primary': 'rgb(90, 98, 108)',
        'secondary': 'rgb(134, 142, 147)',
        'highlight': 'rgb(106, 115, 128)',
        'text': 'rgb(240, 241, 242)',
        'icon': 'rgb(209, 210, 212)',
        'warning': 'rgb(153, 18, 31)',
        'current': 'rgb(0, 122, 204)',
        'syntax_style': 'native',
        'console': 'rgb(0, 0, 0)',
        'canvas': 'black',
    },
    'light': {
        'folder': 'light',
        'background': 'rgb(239, 235, 233)',
        'foreground': 'rgb(214, 208, 206)',
        'primary': 'rgb(188, 184, 181)',
        'secondary': 'rgb(150, 146, 144)',
        'highlight': 'rgb(163, 158, 156)',
        'text': 'rgb(59, 58, 57)',
        'icon': 'rgb(107, 105, 103)',
        'warning': 'rgb(255, 18, 31)',
        'current': 'rgb(253, 240, 148)',
        'syntax_style': 'default',
        'console': 'rgb(255, 255, 255)',
        'canvas': 'white',
    },
}

gradient_pattern = re.compile(r'([vh])gradient\((.+)\)')
darken_pattern = re.compile(r'{{\s?darken\((\w+),?\s?([-\d]+)?\)\s?}}')
lighten_pattern = re.compile(r'{{\s?lighten\((\w+),?\s?([-\d]+)?\)\s?}}')
opacity_pattern = re.compile(r'{{\s?opacity\((\w+),?\s?([-\d]+)?\)\s?}}')


def darken(color: str, percentage=10):
    if color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    ratio = 1 - float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red * ratio), 0), 255)
    green = min(max(int(green * ratio), 0), 255)
    blue = min(max(int(blue * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def lighten(color: str, percentage=10):
    if color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    ratio = float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red + (255 - red) * ratio), 0), 255)
    green = min(max(int(green + (255 - green) * ratio), 0), 255)
    blue = min(max(int(blue + (255 - blue) * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def opacity(color: str, value=255):
    if color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
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


def template(css, **theme):
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
        css = css.replace('{{ %s }}' % k, v)
    return css


def get_theme(name):
    """Get a theme based on its name

    Parameters
    ----------
    name : str
        Name of requested theme.

    Returns
    -------
    theme: dict of str: str
        Theme mapping elements to colors. A copy is created
        so that manipulating this theme can be done without
        side effects.
    """
    if name in _themes:
        theme = _themes[name]
        return theme.copy()
    else:
        raise ValueError(
            trans._(
                "Unrecognized theme {name}. Availabe themes are {themes}",
                deferred=True,
                name=name,
                themes=available_themes(),
            )
        )


def register_theme(name, theme):
    """Get a theme based on its name

    Parameters
    ----------
    name : str
        Name of requested theme.
    theme : dict of str: str
        Theme mapping elements to colors.
    """
    _themes[name] = theme


def available_themes():
    """List available themes

    Returns
    -------
    list of str
        Names of available themes.
    """
    return tuple(_themes)
