"""text_color() function.

Notes
-----
There are many packages that help you write colored or formatted text to
the terminal such as colorit, printy, ansicolors etc. For now we are just
doing it ourselves to avoid a dependency until we decide if this command
line stuff is temporary or not. And because we don't need anything complex.

"""

# These are in "escape code order" from 30 to 37
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
