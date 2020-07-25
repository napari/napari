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
import typing
import textwrap


from numpydoc.docscrape import NumpyDocString

from .components.add_layers_mixin import AddLayersMixin
from .viewer import Viewer

VIEW_DOC = NumpyDocString(Viewer.__doc__)
VIEW_PARAMS = "    " + "\n".join(VIEW_DOC._str_param_list('Parameters')[2:])

DOC = """Create a viewer and add a{n} {name} layer.

{params}

Returns
-------
viewer : :class:`napari.Viewer`
    The newly-created viewer.
"""


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
    return DOC.format(n=n, name=layer_string, params=params)


def _generate_view_function(layer_string: str, method_name: str = None):
    """Autogenerate a ``view_<layer_string>`` method.

    Combines the signatures and docs of ``Viewer`` and
    ``Viewer.add_<layer_string>``.  The returned function is compatible with
    IPython help, introspection, tab completion, and autodocs.

    Here's how it works:
    1. we define `real_func`, which is the (easier to understand) function that
       will do the work of creating a new viewer and adding a layer to it.
    2. we create a **string** (`fakefunc`) that represents how we _would_ have
       typed out the original `view_*` method.
        - `{combo_sig}` is an `inspect.Signature
          <https://docs.python.org/3/library/inspect.html#inspect.Signature>`_
          object (whose string representation is, conveniently, exactly how we
          would have typed the original function).
        - the inner `real_func({inner_sig})` part is basically how we were
          typing the body of the `view_*` functions before, e.g.:
          `(data=data, name=name, scale=scale ...)`
    3. we compile that string into `view_func_code`
    4. finally, we actually evaluate the compiled code and add it to the
       current module's namespace, and provide a `locals()` dict that tells
       python that the function name `real_func` in the `fakefunc` string
       actually corresponds to the `real_func` that we previuosly defined.
       (Note: evaluation at this step is essentially exactly what was
       previously happening when python hit each `def view_*` declaration when
       importing `view_layers.py`)

    Parameters
    ----------
    layer_string : str
        The name of the layer type
    method_name : str
        The name of the method in AddLayersMixin to use, by default will use
        f'add_{layer_string}'

    Returns
    -------
    view_func : Callable
        The complete view_* function
    """
    # name of the corresponding add_* func
    add_string = method_name or f'add_{layer_string}'
    try:
        add_method = getattr(AddLayersMixin, add_string)
    except AttributeError:
        raise AttributeError(f"No Viewer method named '{add_string}'")

    # get signatures of the add_* method and Viewer.__init__
    add_sig = inspect.signature(add_method)
    view_sig = inspect.signature(Viewer)

    # create a new combined signature
    new_params = list(add_sig.parameters.values())[1:]  # [1:] to remove self
    new_params += view_sig.parameters.values()
    new_params = sorted(new_params, key=lambda p: p.kind)
    combo_sig = add_sig.replace(parameters=new_params)

    # define the actual function that will create a new Viewer and add a layer
    def real_func(*args, **kwargs):
        view_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs) if k in view_sig.parameters
        }
        viewer = Viewer(**view_kwargs)
        getattr(viewer, add_string)(*args, **kwargs)
        return viewer

    # make new function string with the combined signature that calls real_func
    fname = f'view_{layer_string}'
    inner_sig = ", ".join([f"{p}={p}" for p in combo_sig.parameters])
    fakefunc = f"def {fname}{combo_sig}:\n    return real_func({inner_sig})\n"

    # evaluate the new function into the current module namespace
    globals = sys.modules[__name__].__dict__
    eval(
        compile(fakefunc, "fakesource", "exec"),
        {
            "real_func": real_func,
            'typing': typing,
            'Union': typing.Union,
            'List': typing.List,
            'NoneType': type(None),
            'Sequence': typing.Sequence,
            'napari': sys.modules.get('napari'),
        },
        globals,
    )
    view_func = globals[fname]  # this is the final function.
    view_func.__doc__ = merge_docs(add_method, layer_string)


for _layer in ('image', 'points', 'labels', 'shapes', 'surface', 'vectors'):
    _generate_view_function(_layer)

_generate_view_function('path', 'open')
