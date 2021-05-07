"""Methods to create a new viewer instance and add a particular layer type.

All functions follow this pattern ,(where <layer_type> is replaced with one
of the layer types):

    def view_<layer_type>(*args, **kwargs):
        # pop all of the viewer kwargs out of kwargs into viewer_kwargs
        viewer = Viewer(**viewer_kwargs)
        add_method = getattr(viewer, f"add_{<layer_type>}")
        add_method(*args, **kwargs)
        return viewer
"""
import inspect

from numpydoc.docscrape import NumpyDocString as _NumpyDocString

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
    params = (
        "\n".join(add_method_doc._str_param_list('Parameters')) + _VIEW_PARAMS
    )
    # this ugliness is because the indentation of the parsed numpydocstring
    # is different for the first parameter :(
    lines = params.splitlines()
    lines = lines[:3] + textwrap.dedent("\n".join(lines[3:])).splitlines()
    params = "\n".join(lines)
    n = 'n' if layer_string.startswith(tuple('aeiou')) else ''
    return _doc_template.format(n=n, layer_string=layer_string, params=params)


def _merge_layer_viewer_sigs_docs(func):
    from .utils.misc import combine_signatures

    layer_string = func.__name__.replace("view_", "")
    if layer_string == 'path':
        add_method = Viewer.open
    else:
        add_method = getattr(Viewer, f'add_{layer_string}')
    func.__doc__ = _merge_docstrings(add_method, layer_string)
    func.__signature__ = combine_signatures(
        add_method, Viewer, return_annotation=Viewer, exclude=('self',)
    )
    func.__annotations__ = {
        **add_method.__annotations__,
        **Viewer.__init__.__annotations__,
        'return': Viewer,
    }
    return func


_viewer_params = inspect.signature(Viewer).parameters


def _make_viewer_add_layer(add_method: str, args, kwargs) -> Viewer:
    """Utility function that creates a viewer, adds a layer, returns viewer."""
    vkwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _viewer_params}
    viewer = Viewer(**vkwargs)
    if 'kwargs' in kwargs:
        kwargs.update(kwargs.pop("kwargs"))
    method = getattr(viewer, add_method)
    method(*args, **kwargs)
    return viewer


@_merge_layer_viewer_sigs_docs
def view_image(*args, **kwargs):
    return _make_viewer_add_layer('add_image', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_labels(*args, **kwargs):
    return _make_viewer_add_layer('add_labels', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_points(*args, **kwargs):
    return _make_viewer_add_layer('add_points', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_shapes(*args, **kwargs):
    return _make_viewer_add_layer('add_shapes', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_surface(*args, **kwargs):
    return _make_viewer_add_layer('add_surface', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_tracks(*args, **kwargs):
    return _make_viewer_add_layer('add_tracks', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_vectors(*args, **kwargs):
    return _make_viewer_add_layer('add_vectors', args, kwargs)


@_merge_layer_viewer_sigs_docs
def view_path(*args, **kwargs):
    return _make_viewer_add_layer('open', args, kwargs)
