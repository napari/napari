"""
This file contains most (all?) permutations of single and dual colors
which a user can try to use as an argument to face_color and edge_color
in the relevant layers. The idea is to parameterize the tests over these
options.

Vispy has a few bugs that we're trying to overcome. First it doesn't
parse lists like [Color('red'), Color('red')]. Second, the color of 'g' and
'green' is different. We're consistent with vispy's behavior ATM, but it
might change in a future release.
"""
from vispy.color import Color, ColorArray
import numpy as np


REDA = (1.0, 0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)
REDF = '#ff0000'
GREENV = Color('green').rgb[1]
GREENA = (0.0, GREENV, 0.0, 1.0)
GREEN = (0.0, GREENV, 0.0)
GREENF = Color('green').hex

one_point = np.array([[10, 20, 30]])
one_point_1d = np.array([10, 11])
all_one_points = [(one_point,), one_point_1d]

two_points = np.array([[10, 10], [20, 20]])
two_threed_points = np.array([[1, 2, 3], [4, 3, 2]])
all_two_points = [two_points, two_threed_points]

single_color_options = [
    RED,
    GREENA,
    'red',
    'g',
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
    np.array([GREEN]),
    np.array([GREENF]),
]

single_colors_as_colorarray = [
    ColorArray(RED),
    ColorArray(GREEN),
    ColorArray(RED),
    ColorArray('#00ff00'),
    ColorArray(GREEN),
    ColorArray(RED),
    ColorArray(RED),
    ColorArray(RED),
    ColorArray(GREEN),
    ColorArray(RED),
    ColorArray(GREEN),
    ColorArray(GREEN),
    ColorArray(GREEN),
    ColorArray(GREEN),
    ColorArray(RED),
    ColorArray(GREEN),
    ColorArray(GREEN),
]

two_color_options = [
    [Color('red'), Color(GREENF)],
    [ColorArray('g'), 'g'],
    (Color('red'), Color(RED)),
    # Doesn't work due to weird interaction between numpy and Color
    # np.array([Color(REDF), Color('red')], dtype='object'),
    # np.array([Color(GREENA), Color(GREENA)], dtype='object'),
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

two_colors_simple = [
    ['red', 'green'],
    ['g', 'g'],
    ['red', 'red'],
    # ['red', 'red'],
    # ['green', 'green'],
    ['red', 'green'],
    ['green', 'green'],
    ['red', 'green'],
    ['red', 'blue'],
    ['red', 'blue'],
    ['green', 'red'],
    # ['green', 'red'],
    ['red', 'green'],
    ['green', 'red'],
]
two_colors_as_colorarray = [ColorArray(color) for color in two_colors_simple]

