"""Methods to create a new viewer instance then add a particular layer type.

All functions follow this pattern, (where <layer_type> is replaced with one
of the layer types, like "image", "points", etc...):

.. code-block:: python

    def view_<layer_type>(*args, **kwargs):
        # ... pop all of the viewer kwargs out of kwargs into viewer_kwargs
        viewer = Viewer(**viewer_kwargs)
        add_method = getattr(viewer, f"add_{<layer_type>}")
        add_method(*args, **kwargs)
        return viewer
"""
import inspect

from numpydoc.docscrape import NumpyDocString as _NumpyDocString

from napari.components.dims import Dims

from .viewer import Viewer

__all__ = [
    'view_image',
    'view_labels',
    'view_path',
    'view_points',
    'view_shapes',
    'view_surface',
    'view_tracks',
    'view_vectors',
    'imshow',
]

_doc_template = """Create a viewer and add a{n} {layer_string} layer.

{params}

Returns
-------
viewer : :class:`napari.Viewer`
    The newly-created viewer.
"""

_VIEW_DOC = _NumpyDocString(Viewer.__doc__)
_VIEW_PARAMS = "    " + "\n".join(_VIEW_DOC._str_param_list('Parameters')[2:])


def _merge_docstrings(add_method, layer_string):
    # create combined docstring with parameters from add_* and Viewer methods
    import textwrap

    add_method_doc = _NumpyDocString(add_method.__doc__)

    # this ugliness is because the indentation of the parsed numpydocstring
    # is different for the first parameter :(
    lines = add_method_doc._str_param_list('Parameters')
    lines = lines[:3] + textwrap.dedent("\n".join(lines[3:])).splitlines()
    params = "\n".join(lines) + "\n" + textwrap.dedent(_VIEW_PARAMS)
    n = 'n' if layer_string.startswith(tuple('aeiou')) else ''
    return _doc_template.format(n=n, layer_string=layer_string, params=params)


def _merge_layer_viewer_sigs_docs(func):
    """Make combined signature, docstrings, and annotations for `func`.

    This is a decorator that combines information from `Viewer.__init__`,
    and one of the `viewer.add_*` methods.  It updates the docstring,
    signature, and type annotations of the decorated function with the merged
    versions.

    Parameters
    ----------
    func : callable
        `view_<layer_type>` function to modify

    Returns
    -------
    func : callable
        The same function, with merged metadata.
    """
    from .utils.misc import _combine_signatures

    # get the `Viewer.add_*` method
    layer_string = func.__name__.replace("view_", "")
    if layer_string == 'path':
        add_method = Viewer.open
    else:
        add_method = getattr(Viewer, f'add_{layer_string}')

    # merge the docstrings of Viewer and viewer.add_*
    func.__doc__ = _merge_docstrings(add_method, layer_string)

    # merge the signatures of Viewer and viewer.add_*
    func.__signature__ = _combine_signatures(
        add_method, Viewer, return_annotation=Viewer, exclude=('self',)
    )

    # merge the __annotations__
    func.__annotations__ = {
        **add_method.__annotations__,
        **Viewer.__init__.__annotations__,
        'return': Viewer,
    }

    # _forwardrefns_ is used by stubgen.py to populate the globalns
    # when evaluate forward references with get_type_hints
    func._forwardrefns_ = {**add_method.__globals__}
    return func


_viewer_params = inspect.signature(Viewer).parameters
_dims_params = Dims.__fields__


def _make_viewer_then(add_method: str, args, kwargs) -> Viewer:
    """Utility function that creates a viewer, adds a layer, returns viewer."""
    vkwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _viewer_params}
    # separate dims kwargs because we want to set those after adding data
    dims_kwargs = {
        k: vkwargs.pop(k) for k in list(vkwargs) if k in _dims_params
    }
    viewer = Viewer(**vkwargs)
    kwargs.update(kwargs.pop("kwargs", {}))
    method = getattr(viewer, add_method)
    method(*args, **kwargs)
    for arg_name, arg_val in dims_kwargs.items():
        setattr(viewer.dims, arg_name, arg_val)
    return viewer


# Each of the following functions will have this pattern:
#
# def view_image(*args, **kwargs):
#     # ... pop all of the viewer kwargs out of kwargs into viewer_kwargs
#     viewer = Viewer(**viewer_kwargs)
#     viewer.add_image(*args, **kwargs)
#     return viewer


@_merge_layer_viewer_sigs_docs
def view_image(*args, **kwargs):
    return _make_viewer_then('add_image', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_labels(*args, **kwargs):
    return _make_viewer_then('add_labels', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_points(*args, **kwargs):
    return _make_viewer_then('add_points', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_shapes(*args, **kwargs):
    return _make_viewer_then('add_shapes', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_surface(*args, **kwargs):
    return _make_viewer_then('add_surface', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_tracks(*args, **kwargs):
    return _make_viewer_then('add_tracks', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_vectors(*args, **kwargs):
    return _make_viewer_then('add_vectors', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_path(*args, **kwargs):
    return _make_viewer_then('open', args, kwargs)


def imshow(data, viewer=None, channel_axis=None, multiscale=False, **kwargs):
    """Add image to viewer and return the layer and the viewer itself

    Args:
        data (_type_): _description_
        viewer (_type_, optional): _description_. Defaults to None.
        channel_axis (_type_, optional): _description_. Defaults to None.
        multiscale (bool, optional): _description_. Defaults to False.

    Returns:
        Viewer: napari.Viewer instance of the viewer
        list: List of newly created layers in Viewer
    """
    vkwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _viewer_params}
    # separate dims kwargs because we want to set those after adding data
    dims_kwargs = {
        k: vkwargs.pop(k) for k in list(vkwargs) if k in _dims_params
    }

    if channel_axis:
        kwargs['channel_axis'] = channel_axis

    if multiscale:
        kwargs['multiscale'] = multiscale

    # create a viewer if one is not provided
    if not viewer:
        viewer = Viewer(**vkwargs)

    # TODO handle multiple layers being created...
    # create the new layer in the viewer
    layer = viewer.add_image(data, **kwargs)

    # TODO: I don't know why this was necessary in _make_viewer_then
    # kwargs.update(kwargs.pop("kwargs", {}))

    # set viewer dims (TODO not sure why this is needed)
    for arg_name, arg_val in dims_kwargs.items():
        setattr(viewer.dims, arg_name, arg_val)

    return viewer, layer
