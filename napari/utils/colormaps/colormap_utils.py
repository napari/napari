import warnings
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import skimage.color as colorconv
from vispy.color import BaseColormap as VispyColormap
from vispy.color import Color, ColorArray, get_colormap, get_colormaps
from vispy.color.colormap import LUT_len

from napari.utils.colormaps.bop_colors import bopd
from napari.utils.colormaps.colormap import Colormap, ColormapInterpolationMode
from napari.utils.colormaps.inverse_colormaps import inverse_cmaps
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.colormaps.vendored import cm
from napari.utils.translations import trans

# All parsable input color types that a user can provide
ColorType = Union[List, Tuple, np.ndarray, str, Color, ColorArray]


ValidColormapArg = Union[
    str,
    ColorType,
    VispyColormap,
    Colormap,
    Tuple[str, VispyColormap],
    Tuple[str, Colormap],
    Dict[str, VispyColormap],
    Dict[str, Colormap],
    Dict,
]


matplotlib_colormaps = _MATPLOTLIB_COLORMAP_NAMES = OrderedDict(
    viridis=trans._p('colormap', 'viridis'),
    magma=trans._p('colormap', 'magma'),
    inferno=trans._p('colormap', 'inferno'),
    plasma=trans._p('colormap', 'plasma'),
    gray=trans._p('colormap', 'gray'),
    gray_r=trans._p('colormap', 'gray r'),
    hsv=trans._p('colormap', 'hsv'),
    turbo=trans._p('colormap', 'turbo'),
    twilight=trans._p('colormap', 'twilight'),
    twilight_shifted=trans._p('colormap', 'twilight shifted'),
    gist_earth=trans._p('colormap', 'gist earth'),
    PiYG=trans._p('colormap', 'PiYG'),
)
_MATPLOTLIB_COLORMAP_NAMES_REVERSE = {
    v: k for k, v in matplotlib_colormaps.items()
}
_VISPY_COLORMAPS_ORIGINAL = _VCO = get_colormaps()
_VISPY_COLORMAPS_TRANSLATIONS = OrderedDict(
    autumn=(trans._p('colormap', 'autumn'), _VCO['autumn']),
    blues=(trans._p('colormap', 'blues'), _VCO['blues']),
    cool=(trans._p('colormap', 'cool'), _VCO['cool']),
    greens=(trans._p('colormap', 'greens'), _VCO['greens']),
    reds=(trans._p('colormap', 'reds'), _VCO['reds']),
    spring=(trans._p('colormap', 'spring'), _VCO['spring']),
    summer=(trans._p('colormap', 'summer'), _VCO['summer']),
    fire=(trans._p('colormap', 'fire'), _VCO['fire']),
    grays=(trans._p('colormap', 'grays'), _VCO['grays']),
    hot=(trans._p('colormap', 'hot'), _VCO['hot']),
    ice=(trans._p('colormap', 'ice'), _VCO['ice']),
    winter=(trans._p('colormap', 'winter'), _VCO['winter']),
    light_blues=(trans._p('colormap', 'light blues'), _VCO['light_blues']),
    orange=(trans._p('colormap', 'orange'), _VCO['orange']),
    viridis=(trans._p('colormap', 'viridis'), _VCO['viridis']),
    coolwarm=(trans._p('colormap', 'coolwarm'), _VCO['coolwarm']),
    PuGr=(trans._p('colormap', 'PuGr'), _VCO['PuGr']),
    GrBu=(trans._p('colormap', 'GrBu'), _VCO['GrBu']),
    GrBu_d=(trans._p('colormap', 'GrBu_d'), _VCO['GrBu_d']),
    RdBu=(trans._p('colormap', 'RdBu'), _VCO['RdBu']),
    cubehelix=(trans._p('colormap', 'cubehelix'), _VCO['cubehelix']),
    single_hue=(trans._p('colormap', 'single hue'), _VCO['single_hue']),
    hsl=(trans._p('colormap', 'hsl'), _VCO['hsl']),
    husl=(trans._p('colormap', 'husl'), _VCO['husl']),
    diverging=(trans._p('colormap', 'diverging'), _VCO['diverging']),
    RdYeBuCy=(trans._p('colormap', 'RdYeBuCy'), _VCO['RdYeBuCy']),
)
_VISPY_COLORMAPS_TRANSLATIONS_REVERSE = {
    v[0]: k for k, v in _VISPY_COLORMAPS_TRANSLATIONS.items()
}
_PRIMARY_COLORS = OrderedDict(
    red=(trans._p('colormap', 'red'), [1.0, 0.0, 0.0]),
    green=(trans._p('colormap', 'green'), [0.0, 1.0, 0.0]),
    blue=(trans._p('colormap', 'blue'), [0.0, 0.0, 1.0]),
    cyan=(trans._p('colormap', 'cyan'), [0.0, 1.0, 1.0]),
    magenta=(trans._p('colormap', 'magenta'), [1.0, 0.0, 1.0]),
    yellow=(trans._p('colormap', 'yellow'), [1.0, 1.0, 0.0]),
)

SIMPLE_COLORMAPS = {
    name: Colormap(
        name=name, display_name=display_name, colors=[[0.0, 0.0, 0.0], color]
    )
    for name, (display_name, color) in _PRIMARY_COLORS.items()
}


# dictionay for bop colormap objects
BOP_COLORMAPS = {
    name: Colormap(value, name=name, display_name=display_name)
    for name, (display_name, value) in bopd.items()
}

INVERSE_COLORMAPS = {
    name: Colormap(value, name=name, display_name=display_name)
    for name, (display_name, value) in inverse_cmaps.items()
}


def _all_rgb():
    """Return all 256**3 valid rgb tuples."""
    base = np.arange(256, dtype=np.uint8)
    r, g, b = np.meshgrid(base, base, base, indexing='ij')
    return np.stack((r, g, b), axis=-1).reshape((-1, 3))


# The following values were precomputed and stored as constants
# here to avoid heavy computation when importing this module.
# The following code can be used to reproduce these values.
#
# rgb_colors = _all_rgb()
# luv_colors = colorconv.rgb2luv(rgb_colors)
# LUVMIN = np.amin(luv_colors, axis=(0,))
# LUVMAX = np.amax(luv_colors, axis=(0,))
# lab_colors = colorconv.rgb2lab(rgb_colors)
# LABMIN = np.amin(lab_colors, axis=(0,))
# LABMAX = np.amax(lab_colors, axis=(0,))

LUVMIN = np.array([0.0, -83.07790815, -134.09790293])
LUVMAX = np.array([100.0, 175.01447356, 107.39905336])
LUVRNG = LUVMAX - LUVMIN

LABMIN = np.array([0.0, -86.18302974, -107.85730021])
LABMAX = np.array([100.0, 98.23305386, 94.47812228])
LABRNG = LABMAX - LABMIN


def convert_vispy_colormap(colormap, name='vispy'):
    """Convert a vispy colormap object to a napari colormap.

    Parameters
    ----------
    colormap : vispy.color.Colormap
        Vispy colormap object that should be converted.
    name : str
        Name of colormap, optional.

    Returns
    -------
    napari.utils.Colormap
    """
    if not isinstance(colormap, VispyColormap):
        raise TypeError(
            trans._(
                'Colormap must be a vispy colormap if passed to from_vispy',
                deferred=True,
            )
        )

    # Not all vispy colormaps have an `_controls`
    # but if they do, we want to use it
    if hasattr(colormap, '_controls'):
        controls = colormap._controls
    else:
        controls = np.zeros((0,))

    # Not all vispy colormaps have an `interpolation`
    # but if they do, we want to use it
    if hasattr(colormap, 'interpolation'):
        interpolation = colormap.interpolation
    else:
        interpolation = 'linear'

    if name in _VISPY_COLORMAPS_TRANSLATIONS:
        display_name, _cmap = _VISPY_COLORMAPS_TRANSLATIONS[name]
    else:
        # Unnamed colormap
        display_name = trans._(name)

    return Colormap(
        name=name,
        display_name=display_name,
        colors=colormap.colors.rgba,
        controls=controls,
        interpolation=interpolation,
    )


def _validate_rgb(colors, *, tolerance=0.0):
    """Return the subset of colors that is in [0, 1] for all channels.

    Parameters
    ----------
    colors : array of float, shape (N, 3)
        Input colors in RGB space.

    Returns
    -------
    filtered_colors : array of float, shape (M, 3), M <= N
        The subset of colors that are in valid RGB space.

    Other Parameters
    ----------------
    tolerance : float, optional
        Values outside of the range by less than ``tolerance`` are allowed and
        clipped to be within the range.

    Examples
    --------
    >>> colors = np.array([[  0. , 1.,  1.  ],
    ...                    [  1.1, 0., -0.03],
    ...                    [  1.2, 1.,  0.5 ]])
    >>> _validate_rgb(colors)
    array([[0., 1., 1.]])
    >>> _validate_rgb(colors, tolerance=0.15)
    array([[0., 1., 1.],
           [1., 0., 0.]])
    """
    lo = 0 - tolerance
    hi = 1 + tolerance
    valid = np.all((colors > lo) & (colors < hi), axis=1)
    filtered_colors = np.clip(colors[valid], 0, 1)
    return filtered_colors


def low_discrepancy_image(image, seed=0.5, margin=1 / 256):
    """Generate a 1d low discrepancy sequence of coordinates.

    Parameters
    ----------
    image : array of int
        A set of labels or label image.
    seed : float
        The seed from which to start the quasirandom sequence.
    margin : float
        Values too close to 0 or 1 will get mapped to the edge of the colormap,
        so we need to offset to a margin slightly inside those values. Since
        the bin size is 1/256 by default, we offset by that amount.

    Returns
    -------
    image_out : array of float
        The set of ``labels`` remapped to [0, 1] quasirandomly.

    """
    phi_mod = 0.6180339887498948482
    image_float = seed + image * phi_mod
    # We now map the floats to the range [0 + margin, 1 - margin]
    image_out = margin + (1 - 2 * margin) * (
        image_float - np.floor(image_float)
    )
    return image_out


def color_dict_to_colormap(colors):
    """
    Generate a color map based on the given color dictionary

    Parameters
    ----------
    colors : dict of int to array of float, shape (4)
        Mapping between labels and color

    Returns
    -------
    colormap : napari.utils.Colormap
        Colormap constructed with provided control colors
    label_color_index : dict of int
        Mapping of Label to color control point within colormap
    """

    MAX_DISTINCT_COLORS = LUT_len

    control_colors = np.unique(list(colors.values()), axis=0)

    if len(control_colors) >= MAX_DISTINCT_COLORS:
        warnings.warn(
            trans._(
                'Label layers with more than {max_distinct_colors} distinct colors will not render correctly. This layer has {distinct_colors}.',
                deferred=True,
                distinct_colors=str(len(control_colors)),
                max_distinct_colors=str(MAX_DISTINCT_COLORS),
            ),
            category=UserWarning,
        )

    colormap = Colormap(
        colors=control_colors, interpolation=ColormapInterpolationMode.ZERO
    )

    control2index = {
        tuple(color): control_point
        for color, control_point in zip(colormap.colors, colormap.controls)
    }

    control_small_delta = 0.5 / len(control_colors)
    label_color_index = {
        label: np.float32(control2index[tuple(color)] + control_small_delta)
        for label, color in colors.items()
    }

    return colormap, label_color_index


def _low_discrepancy(dim, n, seed=0.5):
    """Generate a 1d, 2d, or 3d low discrepancy sequence of coordinates.

    Parameters
    ----------
    dim : one of {1, 2, 3}
        The dimensionality of the sequence.
    n : int
        How many points to generate.
    seed : float or array of float, shape (dim,)
        The seed from which to start the quasirandom sequence.

    Returns
    -------
    pts : array of float, shape (n, dim)
        The sampled points.

    References
    ----------
    ..[1]: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/  # noqa: E501
    """
    phi1 = 1.6180339887498948482
    phi2 = 1.32471795724474602596
    phi3 = 1.22074408460575947536
    seed = np.broadcast_to(seed, (1, dim))
    phi = np.array([phi1, phi2, phi3])
    g = 1 / phi
    n = np.reshape(np.arange(n), (n, 1))
    pts = (seed + (n * g[:dim])) % 1
    return pts


def _color_random(n, *, colorspace='lab', tolerance=0.0, seed=0.5):
    """Generate n random RGB colors uniformly from LAB or LUV space.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    colorspace : str, one of {'lab', 'luv', 'rgb'}
        The colorspace from which to get random colors.
    tolerance : float
        How much margin to allow for out-of-range RGB values (these are
        clipped to be in-range).
    seed : float or array of float, shape (3,)
        Value from which to start the quasirandom sequence.

    Returns
    -------
    rgb : array of float, shape (n, 3)
        RGB colors chosen uniformly at random from given colorspace.
    """
    factor = 6  # about 1/5 of random LUV tuples are inside the space
    expand_factor = 2
    rgb = np.zeros((0, 3))
    while len(rgb) < n:
        random = _low_discrepancy(3, n * factor, seed=seed)
        if colorspace == 'luv':
            raw_rgb = colorconv.luv2rgb(random * LUVRNG + LUVMIN)
        elif colorspace == 'rgb':
            raw_rgb = random
        else:  # 'lab' by default
            # The values in random are in [0, 1], but since the LAB colorspace
            # is not exactly contained in the unit-box, some 3-tuples might not
            # be valid LAB color coordinates. scikit-image handles this by projecting
            # such coordinates into the colorspace, but will also warn when doing this.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message='Color data out of range',
                    category=UserWarning,
                )
                raw_rgb = colorconv.lab2rgb(random * LABRNG + LABMIN)
        rgb = _validate_rgb(raw_rgb, tolerance=tolerance)
        factor *= expand_factor
    return rgb[:n]


def label_colormap(num_colors=256, seed=0.5):
    """Produce a colormap suitable for use with a given label set.

    Parameters
    ----------
    num_colors : int, optional
        Number of unique colors to use. Default used if not given.
        Colors are in addition to a transparent color 0.
    seed : float or array of float, length 3
        The seed for the random color generator.

    Returns
    -------
    colormap : napari.utils.Colormap
        A colormap for use with labels remapped to [0, 1].

    Notes
    -----
    0 always maps to fully transparent.
    """
    # Starting the control points slightly above 0 and below 1 is necessary
    # to ensure that the background pixel 0 is transparent
    midpoints = np.linspace(0.00001, 1 - 0.00001, num_colors)
    control_points = np.concatenate(([0], midpoints, [1.0]))
    # make sure to add an alpha channel to the colors
    colors = np.concatenate(
        (
            _color_random(num_colors + 1, seed=seed),
            np.full((num_colors + 1, 1), 1),
        ),
        axis=1,
    )
    # Insert alpha at layer 0
    colors[0, :] = 0  # ensure alpha is 0 for label 0

    return Colormap(
        name='label_colormap',
        display_name=trans._p('colormap', 'low discrepancy colors'),
        colors=colors,
        controls=control_points,
        interpolation='zero',
    )


def vispy_or_mpl_colormap(name):
    """Try to get a colormap from vispy, or convert an mpl one to vispy format.

    Parameters
    ----------
    name : str
        The name of the colormap.

    Returns
    -------
    colormap : napari.utils.Colormap
        The found colormap.

    Raises
    ------
    KeyError
        If no colormap with that name is found within vispy or matplotlib.
    """
    if name in _VISPY_COLORMAPS_TRANSLATIONS:
        cmap = get_colormap(name)
        colormap = convert_vispy_colormap(cmap, name=name)
    else:
        try:
            mpl_cmap = getattr(cm, name)
            if name in _MATPLOTLIB_COLORMAP_NAMES:
                display_name = _MATPLOTLIB_COLORMAP_NAMES[name]
            else:
                display_name = name
        except AttributeError as e:
            suggestion = _MATPLOTLIB_COLORMAP_NAMES_REVERSE.get(
                name
            ) or _MATPLOTLIB_COLORMAP_NAMES_REVERSE.get(name)
            if suggestion:
                raise KeyError(
                    trans._(
                        'Colormap "{name}" not found in either vispy or matplotlib but you might want to use "{suggestion}".',
                        deferred=True,
                        name=name,
                        suggestion=suggestion,
                    )
                ) from e
            else:
                colormaps = set(_VISPY_COLORMAPS_ORIGINAL).union(
                    set(_MATPLOTLIB_COLORMAP_NAMES)
                )
                raise KeyError(
                    trans._(
                        'Colormap "{name}" not found in either vispy or matplotlib. Recognized colormaps are: {colormaps}',
                        deferred=True,
                        name=name,
                        colormaps=", ".join(
                            sorted(f'"{cm}"' for cm in colormaps)
                        ),
                    )
                )
        mpl_colors = mpl_cmap(np.linspace(0, 1, 256))
        colormap = Colormap(
            name=name, display_name=display_name, colors=mpl_colors
        )

    return colormap


# A dictionary mapping names to VisPy colormap objects
ALL_COLORMAPS = {
    k: vispy_or_mpl_colormap(k) for k in _MATPLOTLIB_COLORMAP_NAMES
}
ALL_COLORMAPS.update(SIMPLE_COLORMAPS)
ALL_COLORMAPS.update(BOP_COLORMAPS)
ALL_COLORMAPS.update(INVERSE_COLORMAPS)

# ... sorted alphabetically by name
AVAILABLE_COLORMAPS = {
    k: v
    for k, v in sorted(ALL_COLORMAPS.items(), key=lambda cmap: cmap[0].lower())
}
# lock to allow update of AVAILABLE_COLORMAPS in threads
AVAILABLE_COLORMAPS_LOCK = Lock()

# curated colormap sets
# these are selected to look good or at least reasonable when using additive
# blending of multiple channels.
MAGENTA_GREEN = ['magenta', 'green']
RGB = ['red', 'green', 'blue']
CYMRGB = ['cyan', 'yellow', 'magenta', 'red', 'green', 'blue']


def _increment_unnamed_colormap(
    existing: List[str], name: str = '[unnamed colormap]'
) -> Tuple[str, str]:
    """Increment name for unnamed colormap.

    Parameters
    ----------
    existing : list of str
        Names of existing colormaps.
    name : str, optional
        Name of colormap to be incremented. by default '[unnamed colormap]'

    Returns
    -------
    name : str
        Name of colormap after incrementing.
    display_name : str
        Display name of colormap after incrementing.
    """
    display_name = trans._('[unnamed colormap]')
    if name == '[unnamed colormap]':
        past_names = [n for n in existing if n.startswith('[unnamed colormap')]
        name = f'[unnamed colormap {len(past_names)}]'
        display_name = trans._(
            "[unnamed colormap {number}]",
            number=len(past_names),
        )

    return name, display_name


def ensure_colormap(colormap: ValidColormapArg) -> Colormap:
    """Accept any valid colormap argument, and return Colormap, or raise.

    Adds any new colormaps to AVAILABLE_COLORMAPS in the process, except
    for custom unnamed colormaps created from color values.

    Parameters
    ----------
    colormap : ValidColormapArg
        See ValidColormapArg for supported input types.

    Returns
    -------
    Colormap

    Warns
    -----
    UserWarning
        If ``colormap`` is not a valid colormap argument type.

    Raises
    ------
    KeyError
        If a string is provided that is not in AVAILABLE_COLORMAPS
    TypeError
        If a tuple is provided and the first element is not a string or the
        second element is not a Colormap.
    TypeError
        If a dict is provided and any of the values are not Colormap instances
        or valid inputs to the Colormap constructor.
    """
    with AVAILABLE_COLORMAPS_LOCK:
        if isinstance(colormap, str):
            name = colormap
            if name not in AVAILABLE_COLORMAPS:
                cmap = vispy_or_mpl_colormap(
                    name
                )  # raises KeyError if not found
                AVAILABLE_COLORMAPS[name] = cmap
        elif isinstance(colormap, Colormap):
            AVAILABLE_COLORMAPS[colormap.name] = colormap
            name = colormap.name
        elif isinstance(colormap, VispyColormap):
            # if a vispy colormap instance is provided, make sure we don't already
            # know about it before adding a new unnamed colormap
            name = None
            for key, val in AVAILABLE_COLORMAPS.items():
                if colormap == val:
                    name = key
                    break

            if not name:
                name, _display_name = _increment_unnamed_colormap(
                    AVAILABLE_COLORMAPS
                )

            # Convert from vispy colormap
            cmap = convert_vispy_colormap(colormap, name=name)
            AVAILABLE_COLORMAPS[name] = cmap

        elif isinstance(colormap, tuple):
            if (
                len(colormap) == 2
                and isinstance(colormap[0], str)
                and isinstance(colormap[1], (VispyColormap, Colormap))
            ):
                name, cmap = colormap
                # Convert from vispy colormap
                if isinstance(cmap, VispyColormap):
                    cmap = convert_vispy_colormap(cmap, name=name)
                else:
                    cmap.name = name
                AVAILABLE_COLORMAPS[name] = cmap
            else:
                colormap = _colormap_from_colors(colormap)
                if colormap is not None:
                    # Return early because we don't have a name for this colormap.
                    return colormap
                raise TypeError(
                    trans._(
                        "When providing a tuple as a colormap argument, either 1) the first element must be a string and the second a Colormap instance 2) or the tuple should be convertible to one or more colors",
                        deferred=True,
                    )
                )

        elif isinstance(colormap, dict):
            if 'colors' in colormap and not (
                isinstance(colormap['colors'], VispyColormap)
                or isinstance(colormap['colors'], Colormap)
            ):
                cmap = Colormap(**colormap)
                name = cmap.name
                AVAILABLE_COLORMAPS[name] = cmap
            elif not all(
                (isinstance(i, VispyColormap) or isinstance(i, Colormap))
                for i in colormap.values()
            ):
                raise TypeError(
                    trans._(
                        "When providing a dict as a colormap, all values must be Colormap instances",
                        deferred=True,
                    )
                )
            else:
                # Convert from vispy colormaps
                for key, cmap in colormap.items():
                    # Convert from vispy colormap
                    if isinstance(cmap, VispyColormap):
                        cmap = convert_vispy_colormap(cmap, name=key)
                    else:
                        cmap.name = key
                    name = key
                    colormap[name] = cmap
                AVAILABLE_COLORMAPS.update(colormap)
                if len(colormap) == 1:
                    name = list(colormap)[0]  # first key in dict
                elif len(colormap) > 1:
                    name = list(colormap.keys())[0]

                    warnings.warn(
                        trans._(
                            "only the first item in a colormap dict is used as an argument",
                            deferred=True,
                        )
                    )
                else:
                    raise ValueError(
                        trans._(
                            "Received an empty dict as a colormap argument.",
                            deferred=True,
                        )
                    )
        else:
            colormap = _colormap_from_colors(colormap)
            if colormap is not None:
                # Return early because we don't have a name for this colormap.
                return colormap

            warnings.warn(
                trans._(
                    'invalid type for colormap: {cm_type}. Must be a {{str, tuple, dict, napari.utils.Colormap, vispy.colors.Colormap}}. Reverting to default',
                    deferred=True,
                    cm_type=type(colormap),
                )
            )

            # Use default colormap
            name = 'gray'

    return AVAILABLE_COLORMAPS[name]


def _colormap_from_colors(colors: ColorType) -> Optional[Colormap]:
    try:
        color_array = transform_color(colors)
    except (ValueError, AttributeError, KeyError):
        return None
    if color_array.shape[0] == 1:
        color_array = np.array([[0, 0, 0, 1], color_array[0]])
    return Colormap(color_array)


def make_default_color_array():
    return np.array([0, 0, 0, 1])


def display_name_to_name(display_name):
    display_name_map = {
        v._display_name: k for k, v in AVAILABLE_COLORMAPS.items()
    }
    return display_name_map.get(
        display_name, list(AVAILABLE_COLORMAPS.keys())[0]
    )
