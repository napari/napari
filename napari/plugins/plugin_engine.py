import importlib
import os
from typing import Union as U, Dict, Set
from types import FunctionType as Function, ModuleType as Module


Action = U[Function, type]  # class inits can be called as functions
# See forward reference syntax here:
# https://stackoverflow.com/questions/38340808/recursive-typing-in-python-3-5
ModuleHierarchy = Dict[str, U[Action, 'ModuleHierarchy']]


def crawl(
    module: U[Module, str], name: str = None, visited: Set[Action] = None
):
    """Crawl a module to get a hierarchy of packages.

    Parameters
    ----------
    module : python Module or string
        The module to crawl. If a string, it should be importable as a module.
    name : string, optional
        A name for the module, if different from the import name. For example,
        one could ``crawl('skimage', 'scikit-image')``.
    visited : set of functions/classes
        Previously seen modules. We only keep objects at their highest level in
        the hierarchy, so that they are visible where library writers intended
        to expose them, rather than at the individual, fully-qualified module
        level.

    Returns
    -------
    hierarchy : dict
        A hierarchy mapping names to either functions, classes, or a
        sub-hierarchy.
    """
    hierarchy = {}
    if visited is None:
        visited = set()
    if type(module) == str:
        module_components = module.split('.')
        if (
            module_components[-1].startswith('test_')
            or module_components[-1].startswith('_')
            or module.startswith('_')
            or 'tests' in module_components
        ):
            return hierarchy
        try:
            module = importlib.import_module(module)
        except ImportError:
            return hierarchy
    for elem_name in dir(module):
        elem = getattr(module, elem_name)
        if (
            not elem_name.startswith('_')
            and type(elem) in [Function, type]
            and elem.__module__ is not None
            and elem.__module__.startswith(module.__name__)
            and elem not in visited
        ):
            hierarchy[elem_name] = elem
            visited.add(elem)
        # note: add elif to insert *variables*, eg. magic structuring elements,
        # into some special namespace, so that plugins can define constants.

    # check whether the module is a package, and if so, recurse into
    # sub-packages. See:
    # https://docs.python.org/3/reference/import.html#package-path-rules
    if hasattr(module, '__path__'):
        for path in module.__path__:
            for filename in os.listdir(path):
                pathname = os.path.join(path, filename)
                modname = None
                if filename.endswith('.py') and filename != '__init__.py':
                    modname = filename[:-3]
                elif os.path.isdir(pathname) and '__init__.py' in os.listdir(
                    pathname
                ):
                    modname = filename
                if modname is not None:
                    fully_qualified_modname = module.__name__ + '.' + modname
                    subhierarchy = crawl(
                        fully_qualified_modname, name=modname, visited=visited
                    )
                    if len(subhierarchy) > 0:
                        hierarchy[modname] = subhierarchy
    return hierarchy
