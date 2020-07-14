"""Patch functions and methods for tracing or anything.

See patch_callables() below.
"""

from importlib import import_module
import types
from typing import Callable, List, Union


class PatchError(Exception):
    """Failed to patch target, config file error?"""

    def __init__(self, message):
        self.message = message


class Module:
    def __init__(self, module_str, module):
        self.module_str = module_str
        self.module = module

    def __str__(self):
        return self.module_str

    def getattr(self, attribute):
        try:
            return getattr(self.module, attribute)
        except AttributeError:
            raise PatchError(
                f"Module {self.module_str} has no attribute {attribute}"
            )


def _import_modules(target_str):
    """Import the leading modules for the target.

    Try loading successively longer segments of the target_str. For example
    with "napari.utils.chunk_loader.ChunkLoader.load_chunk" we will import:
         napari (works)
         napari.utils (works)
         napari.utils.chunk_loader (works)
         napari.utils.chunk_loader.ChunkLoader (fails)

    then we return the tuple (module, "ChunkLoader.load_chunk") where module is
    the loaded napari.utils.chunk_loader module.

    This wastefully re-imports things over and over, but it seems instant.
    """
    parts = target_str.split('.')
    success_str = ""

    for i in range(1, len(target_str)):
        attempt_str = '.'.join(parts[:i])
        try:
            module = import_module(attempt_str)
            success_str = attempt_str
        except ModuleNotFoundError:
            if not success_str:
                # Top-level module no found
                raise PatchError(f"Module not found: {attempt_str}")

            # At least part of the target_str did import, return the
            # inner-most module important and the rest of the target_str
            # that we didn't use.
            attribute_str = '.'.join(parts[i - 1 :])
            return Module(success_str, module), attribute_str


def _patch_attribute(module: Module, attribute_str, patch_func):
    """Patch the function class.method in this module.
    """
    # We expecte attribute_str is <function> or <class>.<method>. We could
    # allow nested classes and functions if we wanted to extend this some.
    if attribute_str.count('.') > 1:
        raise PatchError(f"Module {module} has no attribute {attribute_str}")

    if '.' in attribute_str:
        # Assume attribute_str is <class>.<method>
        class_str, callable_str = attribute_str.split('.')
        parent = module.getattr(class_str)
        parent_str = class_str
    else:
        # Assume attribute_str is <function>.
        class_str = None
        parent = module.module
        parent_str = str(module)
        callable_str = attribute_str

    try:
        getattr(parent, callable_str)
    except AttributeError:
        raise PatchError(
            f"{module}.{parent_str} has no attribute {callable_str}"
        )

    label = (
        callable_str if class_str is None else f"{class_str}.{callable_str}"
    )
    print(f"Patcher: patching {label}")
    patch_func(parent, callable_str, label)


class Target:
    def __init__(self, target_str):
        self.target_str = target_str

    def patch(self, patch_func) -> bool:
        try:
            module, attribute_str = _import_modules(self.target_str)
            _patch_attribute(module, attribute_str, patch_func)
        except PatchError as exc:
            print(f"Patcher: [ERROR] {exc}")
            return False

        return True


# The parent of a callable is a module or a class, class is of type "type".
CallableParent = Union[types.ModuleType, type]

# An example PatchFunction is:
# def _patch_perf_timer(parent, callable: str, label: str) -> None
PatchFunction = Callable[[CallableParent, str, str], None]


def patch_callables(callables: List[str], patch_func: PatchFunction) -> None:
    """Patch the given list of callables.

    Parameters
    ----------
    callables : List[str]
        Patch all of these callables (functions or methods).
    patch_func : PatchFunction
        Called on every callable to patch it.

    Notes
    -----
    The callables list is like this:
    [
        "module.module.ClassName.method_name",
        "module.function_name"
        ...
    ]

    Nested classes and methods not allowed, but support could be added.

    An example patch_func is:

    def _my_patcher(parent: CallableParent, callable: str, label: str) -> None:
        @wrapt.patch_function_wrapper(parent, callable)
        def my_announcer(wrapped, instance, args, kwargs):
            print(f"Announce {label}")
            return wrapped(*args, **kwargs)
    """
    for target_str in callables:
        Target(target_str).patch(patch_func)
