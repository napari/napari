import inspect
import sys
from typing import Callable

from numpydoc.docscrape import NumpyDocString

from .components.add_layers_mixin import AddLayersMixin
from .viewer import Viewer


def _build_view_method(layer_string: str) -> Callable:
    """Autogenerate a ``view_<layer_string>`` method.

    Combines the signatures and docs of ``Viewer`` and
    ``Viewer.add_<layer_string>``.  The returned function is compatible with
    IPython help, introspection, tab completion, and autodocs.


    Parameters
    ----------
    layer_string : str
        The name of the layer type

    Returns
    -------
    view_func : Callable
        The complete view_* function
    """
    add_string = f'add_{layer_string}'  # name of the new function
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
    eval(view_func_code, {"real_func": real_func}, fakeglobals)
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
    setattr(module, f'view_{_layer}', _build_view_method(_layer))
