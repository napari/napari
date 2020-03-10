from typing import Union, List, Set
from types import ModuleType
from pluggy.hooks import HookImpl, _HookCaller

HookOrderType = Union[List[str], List[ModuleType], List[HookImpl]]


def permute_hook_implementations(
    hook_caller: _HookCaller, order: HookOrderType
):
    """Change the call order of hookimplementations for a pluggy HookCaller.

    Pluggy does not allow a built-in way to change the call order after
    instantiation.  hookimpls are called in last-in-first-out order.
    This function accepts the desired call order (a list of plugin names, or
    plugin modules) and reorders the hookcaller accordingly.

    Parameters
    ----------
    hook_caller : pluggy.hooks._HookCaller
        The hook caller to reorder
    order : list
        A list of str, hookimpls, or module_or_class, with the desired
        CALL ORDER of the hook implementations.

    Raises
    ------
    ValueError
        If the 'order' list cannot be interpreted as a list of "plugin_name"
        or "plugin" (module_or_class)
    ValueError
        if 'order' argument has multiple entries for the same hookimpl
    """
    if all(isinstance(o, HookImpl) for o in order):
        attr = None
    elif all(isinstance(o, str) for o in order):
        attr = 'plugin_name'
    elif any(isinstance(o, str) for o in order):
        raise TypeError(
            "order list must be either ALL strings, or ALL modules/classes"
        )
    else:
        attr = 'plugin'

    hookimpls = hook_caller.get_hookimpls()
    if len(order) > len(hookimpls):
        raise ValueError(
            f"too many values ({len(order)} > {len(hookimpls)}) in order."
        )
    if attr:
        hookattrs = [getattr(hookimpl, attr) for hookimpl in hookimpls]
    else:
        hookattrs = hookimpls

    # find the current position of items specified in `order`
    indices = []
    seen: Set[HookOrderType] = set()
    for i in order:
        if i in seen:
            raise ValueError(
                f"'order' argument had multiple entries for hookimpl: {i}"
            )
        seen.add(i)
        try:
            indices.append(hookattrs.index(i))
        except ValueError as e:
            msg = f"Could not find hookimpl '{i}'."
            if attr != 'plugin_name':
                msg += (
                    " If all items in `order` "
                    "argument are not strings, they are assumed to be an "
                    "imported plugin module or class."
                )
            raise ValueError(msg) from e

    # make new arrays for _wrappers and _nonwrappers
    _wrappers = []
    _nonwraps = []
    for i in indices:
        imp = hookimpls[i]
        methods = _wrappers if imp.hookwrapper else _nonwraps
        methods.insert(0, imp)

    # remove items that have been pulled, leaving only items that
    # were not specified in `order` argument
    for i in sorted(indices, reverse=True):
        del hookimpls[i]

    if hookimpls:
        _wrappers = [x for x in hookimpls if x.hookwrapper] + _wrappers
        _nonwraps = [x for x in hookimpls if not x.hookwrapper] + _nonwraps

    hook_caller._wrappers = _wrappers
    hook_caller._nonwrappers = _nonwraps
