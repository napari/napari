"""Methods to create a new viewer instance and add a particular layer type.

This module autogenerates a number of convenience functions, such as
"view_image", or "view_surface", that both instantiate a new viewer instance,
and add a new layer of a specific type to the viewer.  Each convenience
function signature is a merged version of one of the ``Viewer.__init__`` method
and the ``Viewer.add_<layer_type>`` methods.  The final generated functions
follow this pattern
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
import textwrap
import typing

from numpydoc.docscrape import NumpyDocString

from .utils.translations import trans
from .viewer import Viewer

VIEW_DOC = NumpyDocString(Viewer.__doc__)
VIEW_PARAMS = "    " + "\n".join(VIEW_DOC._str_param_list('Parameters')[2:])


def merge_docs(add_method, layer_string):
    # create combined docstring with parameters from add_* and Viewer methods
    add_method_doc = NumpyDocString(add_method.__doc__)
    params = (
        "\n".join(add_method_doc._str_param_list('Parameters')) + VIEW_PARAMS
    )
    # this ugliness is because the indentation of the parsed numpydocstring
    # is different for the first parameter :(
    lines = params.splitlines()
    lines = lines[:3] + textwrap.dedent("\n".join(lines[3:])).splitlines()
    params = "\n".join(lines)
    n = 'n' if layer_string.startswith(tuple('aeiou')) else ''
    return f"""Create a viewer and add a{n} {layer_string} layer.

{params}

Returns
-------
viewer : :class:`napari.Viewer`
    The newly-created viewer.
"""


def _generate_view_function(layer_string: str, method_name: str = None):
    """Autogenerate a ``view_<layer_string>`` method.

    Combines the signatures and docs of ``Viewer`` and
    ``Viewer.add_<layer_string>``.  The returned function is compatible with
    IPython help, introspection, tab completion, and autodocs.

    Here's how it works:
    1. we create a **string** (`fakefunc`) that represents how we _would_ have
       typed out the original `view_*` method.
        - `{combo_sig}` is an `inspect.Signature
          <https://docs.python.org/3/library/inspect.html#inspect.Signature>`_
          object (whose string representation is, conveniently, exactly how we
          would have typed the original function).
        - the inner part is basically how we were typing the body of the
          `view_*` functions before.  That is, ``Viewer`` kwargs go to the
          ``Viewer`` constructor, and everything else goes to the ``add_*``
          method.  Note that we use ``_kw = locals()`` to get a dict of all
          arguments passed to the function.
    2. we compile that string, giving ``__file__`` as the second (``filename``)
       argument, so it appears that the compiled code comes from this file.
    3. finally, we evaluate the compiled code and add it to the
       current module's "globals".  The second argument in the ``eval`` call
       is a locals() namespace required to interpret the evaluated code.
       (Note: evaluation at this step is essentially exactly what was
       previously happening when python hit each `def view_*` declaration when
       importing `view_layers.py`)

    Parameters
    ----------
    layer_string : str
        The name of the layer type
    method_name : str
        The name of the method in Viewer to use, by default will use
        f'add_{layer_string}'

    Returns
    -------
    view_func : Callable
        The complete view_* function
    """
    # name of the corresponding add_* func
    add_string = method_name or f'add_{layer_string}'
    try:
        add_method = getattr(Viewer, add_string)
    except AttributeError:
        raise AttributeError(
            trans._(
                "No Viewer method named '{add_string}'",
                deferred=True,
                add_string=add_string,
            )
        )

    # get signatures of the add_* method and Viewer.__init__
    add_sig = inspect.signature(add_method)
    view_sig = inspect.signature(Viewer)

    # create a new combined signature
    new_params = list(add_sig.parameters.values())[1:]  # [1:] to remove self
    new_params += view_sig.parameters.values()
    new_params = sorted(new_params, key=lambda p: p.kind)
    combo_sig = add_sig.replace(parameters=new_params)

    # make new function string with the combined signature
    fakefunc = f"def view_{layer_string}{combo_sig}:"
    fakefunc += """
        _kw = locals()
        view_kwargs = {
            k: _kw.pop(k) for k in list(_kw) if k in view_sig.parameters
        }
        viewer = napari.Viewer(**view_kwargs)
        if 'kwargs' in _kw:
            _kw.update(_kw.pop("kwargs"))
    """
    fakefunc += f"    viewer.{add_string}(**_kw)\n        return viewer"
    # evaluate the new function into the current module namespace
    globals = sys.modules[__name__].__dict__
    eval(
        compile(fakefunc, __file__, "exec"),
        {
            'typing': typing,
            'view_sig': view_sig,
            'Union': typing.Union,
            'Optional': typing.Optional,
            'List': typing.List,
            'NoneType': type(None),
            'Sequence': typing.Sequence,
            'napari': sys.modules.get('napari'),
        },
        globals,
    )
    view_func = globals[f'view_{layer_string}']  # this is the final function.
    view_func.__doc__ = merge_docs(add_method, layer_string)


for _layer in (
    'image',
    'points',
    'labels',
    'shapes',
    'surface',
    'vectors',
    'tracks',
):
    _generate_view_function(_layer)

_generate_view_function('path', 'open')
