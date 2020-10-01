"""Command related utilities.

Notes
-----
There are many packages for text formatting such as colorit, printy,
ansicolors, termcolor if we decide to use one.
"""


# Colors in "escape code order" from 30 to 37
COLORS = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
]


def _code(color: str) -> str:
    """Return color escape code like '[31m' for red.

    Parameters
    ----------
    color : str
        A supported color like 'red'.

    Returns
    -------
    str
        The formatted string.
    """
    # Escape codes for the 8 main colors go from 30 to 37.
    num_str = str(30 + COLORS.index(color))
    return f"[{num_str}m"


def text_color(string: str, color: str) -> str:
    """Return string formatted with the given color.

    Parameters
    ----------
    string : str
        The string to format.
    color : str
        A supported color such as 'red'

    Returns
    -------
    str
        The formatted string
    """
    return f"\x1b{_code(color)}{string}\x1b[0m"


def highlight(string: str) -> str:
    """Return string highlighted with some accent color.

    We have this function so we can change "the highlight color" in
    one place and all commands will use the new color.

    Parameters
    ----------
    string : str
        The string to return highlighted.

    Returns
    -------
    str
        The colorized string.
    """
    return text_color(string, "cyan")
