"""
This file contains most (all?) permutations of single and dual colors
which a user can try to use as an argument to face_color and edge_color
in the relevant layers. The idea is to parameterize the tests over these
options.

Vispy has a few bugs/limitations that we're trying to overcome. First, it
doesn't parse lists like [Color('red'), Color('red')]. Second, the color of
'g' and 'green' is different. We're consistent with vispy's behavior ATM,
but it might change in a future release.
"""
import numpy as np
from vispy.color import Color, ColorArray

# Apparently there are two types of greens - 'g' is represented by a
# (0, 1, 0) tuple, while 'green' has an approximate value of
# (0, 0.5, 0). This is why these two colors are treated differently
# below.
REDA = (1.0, 0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)
REDF = '#ff0000'
GREENV = Color('green').rgb[1]
GREENA = (0.0, GREENV, 0.0, 1.0)
GREEN = (0.0, GREENV, 0.0)
GREENF = Color('green').hex
REDARR = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
GREENARR = np.array([[0.0, GREENV, 0.0, 1.0]], dtype=np.float32)

single_color_options = [
    RED,
    GREENA,
    'transparent',
    'red',
    'g',
    GREENF,
    '#ffccaa44',
    REDA,
    REDARR[0, :3],
    Color(RED).rgb,
    Color(GREENF).rgba,
    ColorArray('red').rgb,
    ColorArray(GREENA).rgba,
    ColorArray(GREEN).rgb,
    ColorArray([GREENA]).rgba,
    GREENARR,
    REDF,
    np.array([GREEN]),
    np.array([GREENF]),
    None,
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
]

two_color_options = [
    ['red', 'red'],
    ('green', 'red'),
    ['green', '#ff0000'],
    ['green', 'g'],
    ('r' for r in range(2)),
    ['r', 'r'],
    np.array(['r', 'r']),
    np.array([[1, 1, 1, 1], [0, GREENV, 0, 1]]),
    (None, 'green'),
    [GREENARR[0, :3], REDARR[0, :3]],
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
    ['red', 'red'],
    ['red', 'red'],
    ['white', 'green'],
    (None, 'green'),
    ['green', 'red'],
]

two_colors_as_array = [ColorArray(color).rgba for color in two_colors_simple]

invalid_colors = [
    'rr',
    'gf',
    '#gf9gfg',
    '#ff00000',
    '#ff0000ii',
    (-1, 0.0, 0.0, 0.0),
    ('a', 1, 1, 1),
    4,
    (3,),
    (34, 342, 2334, 4343, 32, 0.1, -1),
    np.array([[1, 1, 1, 1, 1]]),
    np.array([[[0, 1, 1, 1]]]),
    ColorArray(['r', 'r']),
    Color('red'),
    (REDARR, GREENARR),
]

warning_colors = [
    np.array([]),
    np.array(['g', 'g'], dtype=object),
    [],
    [[1, 2], [3, 4], [5, 6]],
    np.array([[10], [10], [10], [10]]),
]
