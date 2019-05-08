from enum import Enum


class Symbol(Enum):
    """Symbol: Valid symbol/marker types for the Markers layer.
    The string method returns the valid vispy string.

    """
    ARROW = 'arrow'
    CLOBBER = 'clobber'
    CROSS = 'cross'
    DIAMOND = 'diamond'
    DISC = 'disc'
    HBAR = 'hbar'
    RING = 'ring'
    SQUARE = 'square'
    STAR = 'star'
    TAILED_ARROW = 'tailed_arrow'
    TRIANGLE_DOWN = 'triangle_down'
    TRIANGLE_UP = 'triangle_up'
    VBAR = 'vbar'
    X = 'x'

    def __str__(self):
        """String representation: The string method returns the
        valid vispy symbol string for the Markers visual.
        """
        return self.name.lower()


# Mapping of symbol alias names to the deduplicated name
SYMBOL_ALIAS = {
    'o': Symbol.DISC,
    '*': Symbol.STAR,
    '+': Symbol.CROSS,
    '-': Symbol.HBAR,
    '->': Symbol.TAILED_ARROW,
    '>': Symbol.ARROW,
    '^': Symbol.TRIANGLE_UP,
    'v': Symbol.TRIANGLE_DOWN,
    's': Symbol.SQUARE,
    '|': Symbol.VBAR
}
