import numpy as np

STYLES = {
    "double": {
        "TOP": ("╔", "╤", "╗", "═"),
        "MID": ("╟", "┼", "╢", "─"),
        "BOT": ("╚", "╧", "╝", "═"),
        "V": ("║", "│"),
    },
    "heavy": {
        "TOP": ("┏", "┯", "┓", "━"),
        "MID": ("┠", "┼", "┨", "─"),
        "BOT": ("┗", "┷", "┛", "━"),
        "V": ("┃", "│"),
    },
    "light": {
        "TOP": ("┌", "┬", "┐", "─"),
        "MID": ("├", "┼", "┤", "─"),
        "BOT": ("└", "┴", "┘", "─"),
        "V": ("│", "│"),
    },
    "markdown": {
        "TOP": (" ", " ", " ", " "),
        "MID": ("|", "|", "|", "-"),
        "BOT": (" ", " ", " ", " "),
        "V": ("|", "|"),
    },
}


def table_repr(
    data,
    padding=2,
    ncols=None,
    header=None,
    cell_width=None,
    divide_rows=True,
    style="markdown",
):
    """Pretty string repr of a 2D table."""
    try:
        nrows = len(data)
    except TypeError:
        raise TypeError("data must be a collection")
    if not nrows:
        return ""

    try:
        ncols = ncols or len(data[0])
    except TypeError:
        raise TypeError("data must be a collection")
    except IndexError:
        raise IndexError("data must be a 2D collection of collections")

    _widths = list(data)
    if header:
        _widths.append(list(header))
    _widths = np.array([[len(str(item)) for item in row] for row in _widths])
    cell_widths = _widths.max(0).tolist()

    _style = STYLES[style]
    TOP, MID, BOT, V = _style["TOP"], _style["MID"], _style["BOT"], _style["V"]

    pad = " " * padding
    cell_templates = [
        (pad + "{{:{0}}}" + pad).format(max(cw, 5)) for cw in cell_widths
    ]
    row_template = V[0] + V[1].join(cell_templates) + V[0]

    def _border(left, sep, right, line):
        _cells = [len(ct.format("")) * line for ct in cell_templates]
        return left + sep.join(_cells) + right

    body = [_border(*TOP)]

    if header:
        body.append(row_template.format(*header))
        body.append(_border(*MID))

    for i, row in enumerate(data):
        body.append(row_template.format(*row))
        if divide_rows and i < nrows - 1:
            body.append(_border(*MID))

    body.append(_border(*BOT))
    return "\n".join(body)
