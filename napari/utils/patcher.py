"""Patch functions and methods. Our perfmon system using this to patch in
perf_timers, but this can be used for any type of patching.

See patch_callables() below as the main entrypoint.
"""

from importlib import import_module
import sys
import types
from typing import Callable, Dict, List, Tuple, Union

# The parent of a callable is a module or a class, class is of type "type".
CallableParent = Union[types.ModuleType, type]

# An example PatchFunction is:
# def _patch_perf_timer(parent, callable: str, label: str) -> None
PatchFunction = Callable[[CallableParent, str, str], None]


class PatchError(Exception):
    """Failed to patch target, config file error?"""

    def __init__(self, message):
        self.message = message


class Module:
    """An imported module.
    """

    def __init__(self, module_path, module_obj):
        self.module_path = module_path
        self.module_obj = module_obj

    def __str__(self):
        return self.module_path

    def getattr(self, attribute):
        try:
            return getattr(self.module_obj, attribute)
        except AttributeError:
            raise PatchError(f"Module {self} has no attribute {attribute}")

    def patch_attribute(self, attribute_str, patch_func):
        """Patch the attribute (callable) in this module.
        """
        # We expect attribute_str is <function> or <class>.<method>. We could
        # allow nested classes and functions if we wanted to extend this some.
        if attribute_str.count('.') > 1:
            raise PatchError(f"Nested attribute not found: {attribute_str}")

        if '.' in attribute_str:
            # Assume attribute_str is <class>.<method>
            class_str, callable_str = attribute_str.split('.')
            parent = self.getattr(class_str)
            parent_str = class_str
        else:
            # Assume attribute_str is <function>.
            class_str = None
            parent = self.module_obj
            parent_str = str(self)
            callable_str = attribute_str

        try:
            getattr(parent, callable_str)
        except AttributeError:
            raise PatchError(
                f"Parent {self}.{parent_str} has no attribute {callable_str}"
            )

        label = (
            callable_str
            if class_str is None
            else f"{class_str}.{callable_str}"
        )

        # Patch it with the user-provided patch_func.
        print(f"Patcher: patching {self}.{label}")
        patch_func(parent, callable_str, label)


class Importer:
    def __init__(self):
        # Keep track of modules so we don't re-import them over and over.
        # This is probably not needed because importing a module the second
        # time seems instant. But it's easy to cache these and possibly
        # it's better in some way.
        self.modules: Dict[str, Module] = {}

    def _attempt_import(self, module_path: str) -> Module:
        """
        """
        try:
            return self.modules[module_path]  # Return if already imported.
        except KeyError:
            # Was not already imported so try importing it.
            module_obj = import_module(module_path)
            module = Module(module_path, module_obj)
            self.modules[module_path] = module  # Save for next time.
            return module

    def import_modules(self, target_str: str) -> Tuple[Module, str]:
        """Import the modules in this target string.

        Try loading successively longer segments of the target_str. For example
        with "napari.utils.chunk_loader.ChunkLoader.load_chunk" we will import:
            napari (success)
            napari.utils (success)
            napari.utils.chunk_loader (success)
            napari.utils.chunk_loader.ChunkLoader (fails, not a module)

        Then we return the tuple (module, "ChunkLoader.load_chunk") where module is
        the loaded napari.utils.chunk_loader module.
        """
        parts = target_str.split('.')
        module = None  # Innert-most module imported so far.

        # Progressively try to import longer and longer segments of the path.
        for i in range(1, len(target_str)):
            module_path = '.'.join(parts[:i])
            try:
                module = self._attempt_import(module_path)
            except ModuleNotFoundError:
                if module is None:
                    # The very first top-level module import failed!
                    raise PatchError(f"Module not found: {module_path}")

                # Return the inner-most module we did successfuly import
                # and the rest of the module_path we didn't use.
                attribute_str = '.'.join(parts[i - 1 :])
                return module, attribute_str


class Patcher:
    def __init__(self, patch_func: PatchFunction, stop_on_error=False):
        self._importer = Importer()
        self._patch_func = patch_func
        self._stop_on_error = stop_on_error

    def patch_all(self, callables: List[str]) -> None:
        """Patch all the functions/methods in the list.

        Parameters
        ----------
        callables : List[str]
            List of callables to patch.

        Raises
        ------
        PatchError
            Patch failed, module or callable does not exist.

        """
        for target_str in callables:
            self._patch_target(target_str)

    def _patch_target(self, target_str: str) -> None:
        try:
            module, attribute_str = self._importer.import_modules(target_str)
            module.patch_attribute(attribute_str, self._patch_func)
        except PatchError as exc:
            print(f"Patcher: [ERROR] {exc}")
            if self._stop_on_error:
                sys.exit(1)


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

    import wrapt
    def _my_patcher(parent: CallableParent, callable: str, label: str) -> None:
        @wrapt.patch_function_wrapper(parent, callable)
        def my_announcer(wrapped, instance, args, kwargs):
            print(f"Announce {label}")
            return wrapped(*args, **kwargs)

    Stop on error is nice so the user knows they have a mistake in their
    config file. But if you run different branches with the same config,
    you'd kind of like it to ignore errors, since some things might
    not exist in one of the branches. So not sure which is best.
    """
    Patcher(patch_func, stop_on_error=False).patch_all(callables)
