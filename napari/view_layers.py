"""Methods to create a new viewer instance and add a particular layer type.

This module autogenerates a number of convenience functions, such as
"view_image", or "view_surface", that both instantiate a new viewer instance,
and add a new layer of a specific type to the viewer.  Each convenience
function signature is essentially just a merged version of one of the
``Viewer.add_<layer_type>`` methods, along with the signature of the
``Viewer.__init__`` method.  The final generated functions follow this pattern
(where <layer_type> is replaced with one of the layer types):

    def view_<layer_type>(*args, **kwargs):
        # pop all of the viewer kwargs out of kwargs into viewer_kwargs
        viewer = Viewer(**viewer_kwargs)
        add_method = getattr(viewer, f"add_{<layer_type>}")
        add_method(*args, **kwargs)
        return viewer

Note however: the real function signatures and documentation are maintained in
the final functions, along with introspection, and tab autocompletion, etc...
"""
import inspect
import sys
from typing import Callable

from numpydoc.docscrape import NumpyDocString

from .components.add_layers_mixin import AddLayersMixin
from .viewer import Viewer


def _build_view_function(layer_string: str) -> Callable:
    """Autogenerate a ``view_<layer_string>`` method.

    Combines the signatures and docs of ``Viewer`` and
    ``Viewer.add_<layer_string>``.  The returned function is compatible with
    IPython help, introspection, tab completion, and autodocs.

    Here's how it works:
    1. we define `real_func`, which is the (easier to understand) function that
       will do the work of creating a new viewer and adding a layer to it.
    2. we create a **string** (`fakefunc`) that represents how we _would_ have
       typed out the original `view_*` method.
        - `{combo_sig}` is an
          `inspect.Signature <https://docs.python.org/3/library/inspect.html#inspect.Signature>`_
          object (whose string representation is, conveniently, exactly how we
          would have typed the original function).
        - the inner `real_func({inner_sig})` part is basically how we were
          typing the body of the `view_*` functions before, e.g.:
          `(data=data, name=name, scale=scale ...)`
    3. we compile that string into `view_func_code`
    4. finally, we actually evaluate the compiled code in a (safe) empty
       namespace, and provide a `locals()` dict that tells python that the
       function name `real_func` in the `fakefunc` string actually corresponds
       to the `real_func` that we defined on line 66.   (Note: evaluation at
       this step is essentially exactly what was previously happening when
       python hit each `def view_*` declaration when importing
       `view_layers.py`)

    Parameters
    ----------
    layer_string : str
        The name of the layer type

    Returns
    -------
    view_func : Callable
        The complete view_* function
    """
    add_string = f'add_{layer_string}'  # name of the corresponding add_* func
    try:
        add_method = getattr(AddLayersMixin, add_string)
    except AttributeError:
        raise AttributeError(f"No Viewer method named '{add_string}'")

    # get signatures of the add_* method and Viewer.__init__
    add_sig = inspect.signature(add_method)
    viewer_sig = inspect.signature(Viewer)

    # create a new combined signature
    new_params = list(add_sig.parameters.values())[1:]  # [1:] to remove self
    new_params += [
        p.replace(kind=p.KEYWORD_ONLY) for p in viewer_sig.parameters.values()
    ]
    combo_sig = add_sig.replace(parameters=new_params)

    # define the actual function that will create a new Viewer and add a layer
    def real_func(*args, **kwargs):
        view_kwargs = {}
        for key in list(kwargs.keys()):
            if key in viewer_sig.parameters:
                view_kwargs[key] = kwargs.pop(key)
        viewer = Viewer(**view_kwargs)
        getattr(viewer, add_string)(*args, **kwargs)
        return viewer

    inner_sig = ", ".join([f"{p}={p}" for p in combo_sig.parameters])

    # compile a new function with the combined signature that calls real_func
    fakefunc = f"def func{combo_sig}:\n    return real_func({inner_sig})\n"
    view_func_code = compile(fakefunc, "fakesource", "exec")

    # evaluate the new function in a fake namespace and extract it
    fakeglobals = {}
    import typing
    import napari

    eval(
        view_func_code,
        {
            "real_func": real_func,
            'typing': typing,
            'Union': typing.Union,
            'List': typing.List,
            'napari': napari,
        },
        fakeglobals,
    )
    view_func = fakeglobals["func"]  # this is the final function.

    # create combined docstring with parameters from add_* and Viewer methods
    add_method_doc = NumpyDocString(add_method.__doc__)
    viewer_doc = NumpyDocString(Viewer.__doc__)
    n = 'n' if layer_string.startswith(tuple('aeiou')) else ''
    new_doc = f"Create a viewer and add a{n} {layer_string} layer.\n\n"
    new_doc += "\n".join(add_method_doc._str_param_list('Parameters'))
    new_doc += "    " + "\n".join(viewer_doc._str_param_list('Parameters')[2:])
    new_doc += (
        "\nReturns\n-------\n"
        "viewer : :class:`napari.Viewer`\n"
        "    The newly-created viewer."
    )

    view_func.__doc__ = new_doc

    return view_func


module = sys.modules[__name__]
for _layer in ['image', 'points', 'labels', 'shapes', 'surface', 'vectors']:
    setattr(module, f'view_{_layer}', _build_view_function(_layer))


def view_path(
    path,
    *,
    stack=False,
    plugin=None,
    layer_type=None,
    title='napari',
    ndisplay=2,
    order=None,
    axis_labels=None,
    show=True,
    **kwargs,
):
    """Create a viewer and add a layer whose type will be determined by path.

    Parameters
    ----------

    path : str or list of str
        A filepath, directory, or URL (or a list of any) to open.
    stack : bool, optional
        If a list of strings is passed and ``stack`` is ``True``, then the
        entire list will be passed to plugins.  It is then up to individual
        plugins to know how to handle a list of paths.  If ``stack`` is
        ``False``, then the ``path`` list is broken up and passed to plugin
        readers one by one.  by default False.
    plugin : str, optional
        Name of a plugin to use.  If provided, will force ``path`` to be
        read with the specified ``plugin``.  If the requested plugin cannot
        read ``path``, an exception will be raised.
    layer_type : str, optional
        If provided, will force data read from ``path`` to be passed to the
        corresponding ``add_<layer_type>`` method (along with any
        additional) ``kwargs`` provided to this function.  This *may*
        result in exceptions if the data returned from the path is not
        compatible with the layer_type.
    title : string, optional
        The title of the viewer window. by default 'napari'
    ndisplay : {2, 3}, optional
        Number of displayed dimensions, by default 2
    order : tuple of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3. by default None
    axis_labels : list of str, optional
        Dimension names. by default None
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.
    **kwargs
        All other keyword arguments will be passed on to the respective
        ``add_layer`` method.

    Returns
    -------
    viewer : :class:`napari.Viewer`
        The newly-created viewer.
    """
    viewer = Viewer(
        title=title,
        ndisplay=ndisplay,
        order=order,
        axis_labels=axis_labels,
        show=show,
    )
    viewer.open(
        path=path, stack=stack, plugin=plugin, layer_type=layer_type, **kwargs
    )
    return viewer
