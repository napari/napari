"""Miscellaneous utility functions.
"""
from enum import Enum
import re
import inspect
import itertools
from scipy import ndimage as ndi
from numpydoc.docscrape import FunctionDoc

import numpy as np
import wrapt


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


def is_iterable(arg, color=False):
    """Determine if a single argument is an iterable. If a color is being
    provided and the argument is a 1-D array of length 3 or 4 then the input
    is taken to not be iterable.
    """
    if arg is None:
        return False
    elif type(arg) is str:
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


def is_rgb(shape):
    """If last dim is 3 or 4 assume image is rgb.
    """
    ndim = len(shape)
    last_dim = shape[-1]

    if ndim > 2 and last_dim < 5:
        return True
    else:
        return False


def is_pyramid(data):
    """If shape of arrays along first axis is strictly decreasing.
    """
    size = np.array([np.prod(d.shape, dtype=np.uint64) for d in data])
    if len(size) > 1:
        return np.all(size[:-1] > size[1:])
    else:
        return False


def trim_pyramid(pyramid):
    """Trim very small arrays of top of pyramid.

    Parameters
    ----------
    pyramid : list of array
        Pyramid data

    Returns
    -------
    trimmed : list of array
        Trimmed pyramid data
    """
    keep = [np.any(np.greater_equal(p.shape, 2 ** 6 - 1)) for p in pyramid]
    if np.sum(keep) >= 2:
        return [p for k, p in zip(keep, pyramid) if k]
    else:
        return pyramid[:2]


def should_be_pyramid(shape):
    """Check if any data axes needs to be pyramidified

    Parameters
    ----------
    shape : tuple of int
        Shape of data to be tested

    Returns
    -------
    pyr_axes : tuple of bool
        True wherever an axis exceeds the pyramid threshold.
    """
    return np.log2(shape) >= 13


def get_pyramid_and_rgb(data, pyramid=None, rgb=None):
    """Check if data is or needs to be a pyramid and make one if needed.

    Parameters
    ----------
    data : array, list, or tuple
        Data to be checked if pyramid or if needs to be turned into a pyramid.
    pyramid : bool, optional
        Value that can force data to be considered as a pyramid or not,
        otherwise computed.
    rgb : bool, optional
        Value that can force data to be considered as a rgb, otherwise
        computed.

    Returns
    -------
    ndim : int
        Dimensionality of the data.
    rgb : bool
        If data is rgb.
    pyramid : bool
        If data is a pyramid or a pyramid has been generated.
    data_pyramid : list or None
        If None then data is not and does not need to be a pyramid. Otherwise
        is a list of arrays where each array is a level of the pyramid.
    """
    # Determine if data currently is a pyramid
    currently_pyramid = is_pyramid(data)
    if currently_pyramid:
        shapes = [d.shape for d in data]
        init_shape = shapes[0]
    else:
        init_shape = data.shape

    # Determine if rgb, and determine dimensionality
    if rgb is False:
        pass
    else:
        # If rgb is True or None then guess if rgb
        # allowed or not, and if allowed set it to be True
        rgb_guess = is_rgb(init_shape)
        if rgb and rgb_guess is False:
            raise ValueError(
                "Non rgb or rgba data was passed, but rgb data was"
                " requested."
            )
        else:
            rgb = rgb_guess

    if rgb:
        ndim = len(init_shape) - 1
    else:
        ndim = len(init_shape)

    if pyramid is False:
        if currently_pyramid:
            raise ValueError(
                "Non pyramided data was requested, but pyramid"
                " data was passed"
            )
        else:
            data_pyramid = None
    else:
        if currently_pyramid:
            data_pyramid = trim_pyramid(data)
            pyramid = True
        else:
            # Guess if data should be pyramid or if a pyramid was requested
            if pyramid:
                pyr_axes = [True] * ndim
            else:
                pyr_axes = should_be_pyramid(data.shape)

            if np.any(pyr_axes):
                pyramid = True
                # Set axes to be downsampled to have a factor of 2
                downscale = np.ones(len(data.shape))
                downscale[pyr_axes] = 2
                largest = np.min(np.array(data.shape)[pyr_axes])
                # Determine number of downsample steps needed
                max_layer = np.floor(np.log2(largest) - 9).astype(int)
                data_pyramid = fast_pyramid(
                    data, downscale=downscale, max_layer=max_layer
                )
                data_pyramid = trim_pyramid(data_pyramid)
            else:
                data_pyramid = None
                pyramid = False

    return ndim, rgb, pyramid, data_pyramid


def fast_pyramid(data, downscale=2, max_layer=None):
    """Compute fast image pyramid.

    In the interest of speed this method subsamples, rather than downsamples,
    the input image.

    Parameters
    ----------
    data : array
        Data from which pyramid is to be generated.
    downscale : int or list
        Factor to downscale each step of the pyramid by. If a list, one value
        must be provided for every axis of the array.
    max_layer : int, optional
        The maximum number of layers of the pyramid to be created.

    Returns
    -------
    pyramid : list
        List of arrays where each array is a level of the generated pyramid.
    """

    if max_layer is None:
        max_layer = np.floor(np.log2(np.max(data.shape))).astype(int) + 1

    zoom_factor = np.divide(1, downscale)

    pyramid = [data]
    for i in range(max_layer - 1):
        pyramid.append(
            ndi.zoom(pyramid[i], zoom_factor, prefilter=False, order=0)
        )
    return pyramid


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
    if np.prod(data.shape) > 1e6:
        # If data is very large take the average of the top, bottom, and
        # middle slices
        bottom_plane_idx = (0,) * (data.ndim - 2)
        middle_plane_idx = tuple(s // 2 for s in data.shape[:-2])
        top_plane_idx = tuple(s - 1 for s in data.shape[:-2])
        idxs = [bottom_plane_idx, middle_plane_idx, top_plane_idx]
        reduced_data = [
            [np.max(data[idx]) for idx in idxs],
            [np.min(data[idx]) for idx in idxs],
        ]
    else:
        reduced_data = data

    min_val = np.min(reduced_data)
    max_val = np.max(reduced_data)

    if min_val == max_val:
        min_val = 0
        max_val = 1
    return [float(min_val), float(max_val)]


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


def segment_normal(a, b, p=(0, 0, 1)):
    """Determines the unit normal of the vector from a to b.

    Parameters
    ----------
    a : np.ndarray
        Length 2 array of first point or Nx2 array of points
    b : np.ndarray
        Length 2 array of second point or Nx2 array of points
    p : 3-tuple, optional
        orthogonal vector for segment calculation in 3D.

    Returns
    -------
    unit_norm : np.ndarray
        Length the unit normal of the vector from a to b. If a == b,
        then returns [0, 0] or Nx2 array of vectors
    """
    d = b - a

    if d.ndim == 1:
        if len(d) == 2:
            normal = np.array([d[1], -d[0]])
        else:
            normal = np.cross(d, p)
        norm = np.linalg.norm(normal)
        if norm == 0:
            norm = 1
    else:
        if d.shape[1] == 2:
            normal = np.stack([d[:, 1], -d[:, 0]], axis=0).transpose(1, 0)
        else:
            normal = np.cross(d, p)

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
        if (
            self._default is not inspect._empty
            or kind == inspect._KEYWORD_ONLY
        ):
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
            #     # separator to the parameters list
            #     # ("foo(arg1, *, arg2)" case)
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


class ReadOnlyWrapper(wrapt.ObjectProxy):
    """
    Disable item and attribute setting with the exception of  ``__wrapped__``.
    """

    def __setattr__(self, name, val):
        if name != '__wrapped__':
            raise TypeError(f'cannot set attribute {name}')
        super().__setattr__(name, val)

    def __setitem__(self, name, val):
        raise TypeError(f'cannot set item {name}')


def mouse_press_callbacks(obj, event):
    """Run mouse press callbacks on either layer or viewer object.

    Note that drag callbacks should have the following form::
        def hello_world(layer, event):
            "dragging"
            # on press
            print('hello world!')
            yield

            # on move
            while event.type == 'mouse_move':
                print(event.pos)
                yield

            # on release
            print('goodbye world ;(')

    Parameters
    ---------
    obj : napari.components.ViewerModel or napar.layers.Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event
    """
    # iterate through drag callback functions
    for mouse_drag_func in obj.mouse_drag_callbacks:
        # exectute function to run press event code
        gen = mouse_drag_func(obj, event)
        # if function returns a generator then try to iterate it
        if inspect.isgeneratorfunction(mouse_drag_func):
            try:
                next(gen)
                # now store iterated genenerator
                obj._mouse_drag_gen[mouse_drag_func] = gen
                # and now store event that initially triggered the press
                obj._persisted_mouse_event[gen] = event
            except StopIteration:
                pass


def mouse_move_callbacks(obj, event):
    """Run mouse move callbacks on either layer or viewer object.

    Note that drag callbacks should have the following form::
        def hello_world(layer, event):
            "dragging"
            # on press
            print('hello world!')
            yield

            # on move
            while event.type == 'mouse_move':
                print(event.pos)
                yield

            # on release
            print('goodbye world ;(')

    Parameters
    ---------
    obj : napari.components.ViewerModel or napar.layers.Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event
    """
    if not event.is_dragging:
        # if not dragging simply call the mouse move callbacks
        for mouse_move_func in obj.mouse_move_callbacks:
            mouse_move_func(obj, event)

    # for each drag callback get the current generator
    for func, gen in tuple(obj._mouse_drag_gen.items()):
        # save the event current event
        obj._persisted_mouse_event[gen].__wrapped__ = event
        try:
            # try to advance the generator
            next(gen)
        except StopIteration:
            # If done deleted the generator and stored event
            del obj._mouse_drag_gen[func]
            del obj._persisted_mouse_event[gen]


def mouse_release_callbacks(obj, event):
    """Run mouse release callbacks on either layer or viewer object.

    Note that drag callbacks should have the following form::
        def hello_world(layer, event):
            "dragging"
            # on press
            print('hello world!')
            yield

            # on move
            while event.type == 'mouse_move':
                print(event.pos)
                yield

            # on release
            print('goodbye world ;(')

    Parameters
    ---------
    obj : napari.components.ViewerModel or napar.layers.Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event
    """
    for func, gen in tuple(obj._mouse_drag_gen.items()):
        obj._persisted_mouse_event[gen].__wrapped__ = event
        try:
            # Run last part of the function to trigger release event
            next(gen)
        except StopIteration:
            pass
        # Finally delete the generator and stored event
        del obj._mouse_drag_gen[func]
        del obj._persisted_mouse_event[gen]


def get_keybindings_summary(keymap):
    """Get summary of keybindings in keymap.

    Parameters
    ---------
    keymap : dict
        Dictionary of keybindings.

    Returns
    ---------
    keybindings_str : str
        String with summary of all keybindings and their functions.
    """
    keybindings_str = ''
    for key in keymap:
        func_str = f'<b> {key}</b>: {get_function_summary(keymap[key])}<br>'
        keybindings_str += func_str

    return keybindings_str


def get_function_summary(func):
    """Get summary of doc string of function."""
    doc = FunctionDoc(func)
    summary = ''
    summary += doc['Signature']
    for s in doc['Summary']:
        summary += '<br>&nbsp;&nbsp;&nbsp;&nbsp;' + s
    return summary
