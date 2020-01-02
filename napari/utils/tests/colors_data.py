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

# Apparently there a re two types of greens - 'g' is represented by a
# (0, 1, 0) tuple, while 'green' has an approximate value of
# (0, 0.5, 0). This is why these two colors are treated differently
# below
REDA = (1.0, 0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)
REDF = '#ff0000'
GREENV = Color('green').rgb[1]
GREENA = (0.0, GREENV, 0.0, 1.0)
GREEN = (0.0, GREENV, 0.0)
GREENF = Color('green').hex

single_color_options = [
    RED,
    GREENA,
    'transparent',
    'red',
    'g',
    GREENF,
    '#ffccaa44',
    Color('red'),
    Color(REDA),
    Color(RED).rgb,
    Color(GREENF).rgba,
    ColorArray('red'),
    ColorArray(GREENA).rgba,
    ColorArray(GREEN).rgb,
    ColorArray([GREENA]).rgba,
    ColorArray([GREEN]),
    ColorArray(REDF),
    np.array([GREEN]),
    np.array([GREENF]),
    None,
    '',
]

single_colors_as_array = [
    ColorArray(RED).rgba,
    ColorArray(GREEN).rgba,
    ColorArray((0.0, 0.0, 0.0, 0.0)).rgba,
    ColorArray(RED).rgba,
    ColorArray('#00ff00').rgba,
    ColorArray(GREEN).rgba,
    ColorArray('#ffccaa44').rgba,
    ColorArray(RED).rgba,
    ColorArray(RED).rgba,
    ColorArray(RED).rgba,
    ColorArray(GREEN).rgba,
    ColorArray(RED).rgba,
    ColorArray(GREEN).rgba,
    ColorArray(GREEN).rgba,
    ColorArray(GREEN).rgba,
    ColorArray(GREEN).rgba,
    ColorArray(RED).rgba,
    ColorArray(GREEN).rgba,
    ColorArray(GREEN).rgba,
    np.zeros((1, 4), dtype=np.float32),
    np.zeros((1, 4), dtype=np.float32),
]

two_color_options = [
    ['red', 'red'],
    ('green', 'red'),
    ['green', '#ff0000'],
    ['green', 'g'],
    ('r' for r in range(2)),
    (Color('red'), Color('green')),
    [ColorArray('green'), ColorArray('red')],
    ColorArray(['r', 'r']),
    np.array(['r', 'r']),
    np.array([[1, 1, 1, 1], [0, GREENV, 0, 1]]),
    np.array([[3, 3, 3, 3], [0, 0, 0, 1]]),
    (None, 'green'),
]
# Some of the options below are commented out. When the bugs with
# vispy described above are resolved, we can uncomment the lines
# below as well.
two_colors_simple = [
    ['red', 'red'],
    ['green', 'red'],
    ['green', 'red'],
    ['green', 'g'],
    ['red', 'red'],
    ['red', 'green'],
    ['green', 'red'],
    ['red', 'red'],
    ['red', 'red'],
    ['white', 'green'],
    ['white', 'k'],
    (None, 'green'),
]

two_colors_as_array = [ColorArray(color).rgba for color in two_colors_simple]

invalid_colors = [
    'rr',
    'gf',
    '#gf9gfg',
    '#ff00000',
    '#ff0000ii',
    (43, 3, 3, 3),  # RGBA color, but not parsed correctly
    (-1, 0.0, 0.0, 0.0),
    ('a', 1, 1, 1),
    4,
    (3,),
    (34, 342, 2334, 4343, 32, 0.1, -1),
    np.array([[1, 1, 1, 1, 1]]),
    np.array([[[0, 1, 1, 1]]]),
]

warning_colors = [
    np.array([]),
    np.array(['g', 'g'], dtype=object),
    [],
    [[1, 2], [3, 4], [5, 6]],
    np.array([[10], [10], [10], [10]]),
]
