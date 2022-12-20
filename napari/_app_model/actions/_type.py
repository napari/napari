from inspect import isgeneratorfunction, signature

from app_model.types import Action
from pydantic import Field


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

    def __init__(self, func):
        if not isgeneratorfunction(func):
            raise TypeError(f"'{func.__name__}' is not a generator function")
        self.func = func
        self.__signature__ = signature(self.func)
        self._gen = None

    def __call__(self, *args, **kwargs):
        if self._gen is None:
            self._gen = self.func(*args, **kwargs)
        try:
            next(self._gen)
        except StopIteration:
            self._gen = None
