"""
This file contains most (all?) permutations of single and dual colors
which a user can try to use as an argument to face_color and edge_color
in the relevant layers. The idea is to parameterize the tests over these
options.
"""
from vispy.color import Color, ColorArray
import numpy as np


REDA = (1.0, 0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)
REDF = '#ff0000'
GREENA = (0.0, 1.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0)
GREENF = '#00ff00'

one_point = np.array([[10, 20, 30]])
one_point_1d = np.array([10, 11])
all_one_points = [one_point, one_point_1d]

two_points = np.array([[10, 10], [20, 20]])
two_threed_points = np.array([[1, 2, 3], [4, 3, 2]])
all_two_points = [two_points, two_threed_points]

single_color_options = [
    'red',
    GREENF,
    Color('red'),
    Color(REDA),
    Color(RED),
    Color(GREENF),
    ColorArray('red'),
    ColorArray(GREENA),
    ColorArray(GREEN),
    ColorArray([GREENA]),
    ColorArray([GREEN]),
    ColorArray(REDF),
]

two_color_options = [
    [Color('red'), Color(GREENF)],
    (Color('red'), Color(RED)),
    np.array([Color(REDF), Color('red')], dtype='object'),
    np.array([Color(GREENA), Color(GREENA)]),
    [REDF, GREENA],
    (GREENA, GREEN),
    ['red', GREENF],
    ('red', 'blue'),
    ColorArray(['red', 'blue']),
    ColorArray([GREEN, RED]),
    # ColorArray([GREENA, RED]),  # doesn't work due to a vispy error
    ColorArray((REDF, GREENA)),
    ColorArray(np.array([GREEN, RED])),
]
