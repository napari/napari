"""Miscellaneous utility functions.
"""
from enum import Enum
import re
import numpy as np
import inspect
import itertools


def str_to_rgb(arg):
    """Convert an rgb string 'rgb(x,y,z)' to a list of ints [x,y,z].
    """
    return list(
        map(int, re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', arg).groups())
    )


def ensure_iterable(arg, color=False):
    """Ensure an argument is an iterable. Useful when an input argument
    can either be a single value or a list. If a color is passed then it
    will be treated specially to determine if it is iterable.
    """
    if is_iterable(arg, color=color):
        return arg
    else:
        return itertools.repeat(arg)


def has_clims(arg):
    """Check if a layer has clims.
    """
    if hasattr(arg, '_qt_controls'):
        if hasattr(arg._qt_controls, 'climSlider'):
            return True
        else:
            return False
    else:
        return False


def is_iterable(arg, color=False):
    """Determine if a single argument is an iterable. If a color is being
    provided and the argument is a 1-D array of length 3 or 4 then the input
    is taken to not be iterable.
    """
    if type(arg) is str:
        return False
    elif np.isscalar(arg):
        return False
    elif color and isinstance(arg, (list, np.ndarray)):
        if np.array(arg).ndim == 1 and (len(arg) == 3 or len(arg) == 4):
            return False
        else:
            return True
    else:
        return True


def is_multichannel(shape):
    """If last dim is 3 or 4 assume image is multichannel.
    """
    ndim = len(shape)
    last_dim = shape[-1]

    if ndim > 2 and last_dim < 5:
        return True
    else:
        return False


def increment_unnamed_colormap(name, names):
    """Increment name for unnamed colormap.

    Parameters
    ----------
    name : str
        Name of colormap to be incremented.
    names : str
        Names of existing colormaps.

    Returns
    -------
    name : str
        Name of colormap after incrementing.
    """
    if name == '[unnamed colormap]':
        past_names = [n for n in names if n.startswith('[unnamed colormap')]
        name = f'[unnamed colormap {len(past_names)}]'
    return name


def calc_data_range(data):
    """Calculate range of data values. If all values are equal return [0, 1].

    Parameters
    -------
    data : array
        Data to calculate range of values over.

    Returns
    -------
    values : list of float
        Range of values.
    """
    min = data.min()
    max = data.max()
    if min == max:
        min = 0
        max = 1
    return [float(min), float(max)]


def compute_max_shape(shapes, max_dims=None):
    """Computes the maximum shape combination from the given shapes.

    Parameters
    ----------
    shapes : iterable of tuple
        Shapes to coombine.
    max_dims : int, optional
        Pre-computed maximum dimensions of the final shape.
        If None, is computed on the fly.

    Returns
    -------
    max_shape : tuple
        Maximum shape combination.
    """
    shapes = tuple(shapes)

    if max_dims is None:
        max_dims = max(len(shape) for shape in shapes)

    max_shape = [0] * max_dims

    for dim in range(max_dims):
        for shape in shapes:
            try:
                dim_len = shape[dim]
            except IndexError:
                pass
            else:
                if dim_len > max_shape[dim]:
                    max_shape[dim] = dim_len
    return tuple(max_shape)


def formatdoc(obj):
    """Substitute globals and locals into an object's docstring."""
    frame = inspect.currentframe().f_back
    try:
        obj.__doc__ = obj.__doc__.format(
            **{**frame.f_globals, **frame.f_locals}
        )
        return obj
    finally:
        del frame


def segment_normal(a, b):
    """Determines the unit normal of the vector from a to b.

    Parameters
    ----------
    a : np.ndarray
        Length 2 array of first point or Nx2 array of points
    b : np.ndarray
        Length 2 array of second point or Nx2 array of points

    Returns
    -------
    unit_norm : np.ndarray
        Length the unit normal of the vector from a to b. If a == b,
        then returns [0, 0] or Nx2 array of vectors
    """
    d = b - a

    if d.ndim == 1:
        normal = np.array([d[1], -d[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            norm = 1
    else:
        normal = np.stack([d[:, 1], -d[:, 0]], axis=0).transpose(1, 0)
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        ind = norm == 0
        norm[ind] = 1
    unit_norm = normal / norm

    return unit_norm


def segment_normal_vector(a, b):
    """Determines the unit normal of the vector from a to b.

    Parameters
    ----------
    a : np.ndarray
        Length 2 array of first point
    b : np.ndarray
        Length 2 array of second point

    Returns
    -------
    unit_norm : np.ndarray
        Length the unit normal of the vector from a to b. If a == b,
        then returns [0, 0]
    """
    d = b - a
    normal = np.array([d[1], -d[0]])
    norm = np.linalg.norm(normal)
    if norm == 0:
        unit_norm = np.array([0, 0])
    else:
        unit_norm = normal / norm
    return unit_norm


def interpolate_coordinates(old_coord, new_coord, brush_size):
    """Interpolates coordinates depending on brush size.

    Useful for ensuring painting is continuous in labels layer.

    Parameters
    ----------
    old_coord : np.ndarray, 1x2
        Last position of cursor.
    new_coord : np.ndarray, 1x2
        Current position of cursor.
    brush_size : float
        Size of brush, which determines spacing of interploation.

    Returns
    ----------
    coords : np.array, Nx2
        List of coordinates to ensure painting is continous
    """
    num_step = round(
        max(abs(np.array(new_coord) - np.array(old_coord))) / brush_size * 4
    )
    coords = [
        np.linspace(old_coord[i], new_coord[i], num=num_step + 1)
        for i in range(len(new_coord))
    ]
    coords = np.stack(coords).T
    if len(coords) > 1:
        coords = coords[1:]

    return coords


class StringEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        """ autonaming function assigns each value its own name as a value
        """
        return name.lower()

    def _missing_(self, value):
        """ function called with provided value does not match any of the class
           member values. This function tries again with an upper case string.
        """
        return self(value.lower())

    def __str__(self):
        """String representation: The string method returns the lowercase
        string of the Enum name
        """
        return self.value


camel_to_snake_pattern = re.compile(r'(.)([A-Z][a-z]+)')


def camel_to_snake(name):
    # https://gist.github.com/jaytaylor/3660565
    return camel_to_snake_pattern.sub(r'\1_\2', name).lower()


class CallDefault(inspect.Parameter):
    def __str__(self):
        """wrap defaults"""
        kind = self.kind
        formatted = self._name

        # Fill in defaults
        if self._default is not inspect._empty:
            formatted = '{}={}'.format(formatted, formatted)

        if kind == inspect._VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == inspect._VAR_KEYWORD:
            formatted = '**' + formatted

        return formatted


class CallSignature(inspect.Signature):
    _parameter_cls = CallDefault

    def __str__(self):
        """do not render separators

        commented code is what was taken out from
        the copy/pasted inspect module code :)
        """
        result = []
        # render_pos_only_separator = False
        # render_kw_only_separator = True
        for param in self.parameters.values():
            formatted = str(param)

            # kind = param.kind

            # if kind == inspect._POSITIONAL_ONLY:
            #     render_pos_only_separator = True
            # elif render_pos_only_separator:
            #     # It's not a positional-only parameter, and the flag
            #     # is set to 'True' (there were pos-only params before.)
            #     result.append('/')
            #     render_pos_only_separator = False

            # if kind == inspect._VAR_POSITIONAL:
            #     # OK, we have an '*args'-like parameter, so we won't need
            #     # a '*' to separate keyword-only arguments
            #     render_kw_only_separator = False
            # elif kind == inspect._KEYWORD_ONLY and render_kw_only_separator:
            #     # We have a keyword-only parameter to render and we haven't
            #     # rendered an '*args'-like parameter before, so add a '*'
            #     # separator to the parameters list ("foo(arg1, *, arg2)" case)
            #     result.append('*')
            #     # This condition should be only triggered once, so
            #     # reset the flag
            #     render_kw_only_separator = False

            result.append(formatted)

        # if render_pos_only_separator:
        #     # There were only positional-only parameters, hence the
        #     # flag was not reset to 'False'
        #     result.append('/')

        rendered = '({})'.format(', '.join(result))

        if self.return_annotation is not inspect._empty:
            anno = inspect.formatannotation(self.return_annotation)
            rendered += ' -> {}'.format(anno)

        return rendered


callsignature = CallSignature.from_callable
