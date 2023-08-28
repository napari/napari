import functools
import inspect
from typing import Callable

from app_model.types import Action
from pydantic import Field

from napari.utils.translations import trans


class RepeatableAction(Action):
    repeatable: bool = Field(
        True,
        description="Whether this command is triggered repeatedly when its keybinding is held down.",
    )


class GeneratorCallback:
    """Wrapper for generator callbacks.

    This takes a generator function and returns a callable object that cycles
    through the generator with successive calls, refreshing the generator
    as-needed.
    """

    def __init__(self, func: Callable):
        if not inspect.isgeneratorfunction(func):
            raise TypeError(f"'{func.__name__}' is not a generator function")
        self.func = func
        # make this callable object look more like func (copy the signature, docstring, etc.)
        functools.update_wrapper(self, func)
        self._gen = None

    def __call__(self, *args, **kwargs):
        if self._gen is None:
            self._gen = self.func(*args, **kwargs)
        try:
            next(self._gen)
        except StopIteration:
            self._gen = None


class AttrRestoreCallback:
    """Wrapper for callbacks that should restore the value of some attribute after running.

    This takes a function and an attribute_name, and turns it into a
    GeneratorCallback that will restore attribute_name on the injected object to
    its previous state.

    See napari.layers.utils.layer_utils.register_layer_attr_action for more info.
    """

    def __init__(self, func: Callable, attribute_name: str):
        sig = inspect.signature(func)
        try:
            first_variable_name = next(iter(sig.parameters))
        except StopIteration:
            raise RuntimeError(
                trans._(
                    "If actions has no arguments there is no way to know what to set the attribute to.",
                    deferred=True,
                ),
            )

        # create a wrapper that stores the previous state of obj.attribute_name
        # and returns a callback to restore it
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if args:
                obj = args[0]
            else:
                obj = kwargs[first_variable_name]
            prev_mode = getattr(obj, attribute_name)
            func(*args, **kwargs)

            def _callback():
                setattr(obj, attribute_name, prev_mode)

            return _callback

        self.attribute_name = attribute_name
        self.func = _wrapper
        self.__signature__ = inspect.signature(_wrapper)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
