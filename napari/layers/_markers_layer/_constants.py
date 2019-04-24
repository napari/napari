from enum import Enum


class Symbols(Enum):
    """Symbols: Valid symbol/marker types for the Markers layer.
    The string method returns the valid vispy string.

    See the Vispy documentation for more details
    on the MarkersVisuals:
        http://vispy.org/visuals.html#vispy.visuals.MarkersVisual
    """
    ARROW = 0
    CLOBBER = 1
    CROSS = 2
    DIAMOND = 3
    DISC = 4
    HBAR = 5
    RING = 6
    SQUARE = 7
    STAR = 8
    TAILED_ARROW = 9
    TRIANGLE_DOWN = 10
    TRIANGLE_UP = 11
    VBAR = 12
    X = 13

    def __str__(self):
        """String representation: The string method returns the
        valid vispy symbol string for the Markers visual.
        """
        return self.name.lower()
