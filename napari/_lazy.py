from importlib import import_module


def install_lazy(module_name, submodules=None, submod_attrs=None):
    """Install lazily loaded submodules, and functions or other attributes.

    Parameters
    ----------
    module_name : str
        Typically use __name__.
    submodules : set
        List of submodules to install.
    submod_attrs : dict
        Dictionary of submodule -> list of attributes / functions.
        These attributes are imported as they are used.

    Returns
    -------
    __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    submodules = set() if submodules is None else set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    __all__ = list(submodules | attr_to_modules.keys())

    def __getattr__(name):
        # this unused import is here to fix a very strange bug.
        # there is some mysterious magical goodness in scipy stats that needs
        # to be imported early.
        # see: https://github.com/napari/napari/issues/925
        # see: https://github.com/napari/napari/issues/1347
        from scipy import stats  # noqa: F401

        if name in submodules:
            return import_module(f'{module_name}.{name}')
        elif name in attr_to_modules:
            try:
                submod = import_module(
                    f'{module_name}.{attr_to_modules[name]}'
                )
            except AttributeError as er:
                # if we want any useful error message to show
                # (besides just "cannot import name...") then we need raise anything
                # BUT an attribute error here, because the __getattr__ protocol will
                # swallow that error.
                raise ImportError(
                    f'Failed to import {attr_to_modules[name]} from {module_name}. '
                    'See cause above'
                ) from er
            # this is where we allow an attribute error to be raised.
            return getattr(submod, name)
        else:
            raise AttributeError(f'No {module_name} attribute {name}')

    def __dir__():
        return __all__

    return __getattr__, __dir__, __all__
