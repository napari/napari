"""This module contains the colormap dictionaries for inverse lookup tables taken
from https://github.com/cleterrier/ChrisLUTs. To make it compatible with napari's
colormap classes, all the values in the colormap are normalized (divide by 255).
"""

from ..translations import trans

I_Bordeaux = [[1, 1, 1], [204 / 255, 0, 51 / 255]]
I_Blue = [[1, 1, 1], [0, 51 / 255, 204 / 255]]
I_Forest = [[1, 1, 1], [0, 153 / 255, 0]]
I_Orange = [[1, 1, 1], [1, 117 / 255, 0]]
I_Purple = [[1, 1, 1], [117 / 255, 0, 1]]

inverse_LUT = {
    "I Bordeaux": (trans._("I Bordeaux"), I_Bordeaux),
    "I Blue": (trans._("I Blue"), I_Blue),
    "I Forest": (trans._("I Forest"), I_Forest),
    "I Orange": (trans._("I Orange"), I_Orange),
    "I_Purple": (trans._("I Purple"), I_Purple),
}
