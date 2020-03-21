from typing import Union, List
from pluggy.hooks import HookImpl, _HookCaller

HookOrderType = Union[List[str], List[HookImpl]]


def _guess_order_type(order: HookOrderType) -> str:
    """Guess which of the two HookOrderType lists ``order`` is, or raise.

    Parameters
    ----------
    order : list
        A list of str (plugin names), or ``HookImpl`` instances,

    Returns
    -------
    str
        a string representation of the type of objects in the order list.
        ('plugin_name' or 'hook_implementation_instance')

    Raises
    ------
    TypeError
        If the list is neither entirely strings nor entirely HookImpl instances
    """
    if all(isinstance(o, HookImpl) for o in order):
        return 'hook_implementation_instance'
    if all(isinstance(o, str) for o in order):
        return 'plugin_name'
    raise TypeError(
        "``order`` must be composed of either ALL str (plugin names), "
        "or ALL HookImpl instances"
    )


def permute_hook_implementations(
    hook_caller: _HookCaller, order: HookOrderType
):
    """Rearrange the call order of hook implementations in a pluggy HookCaller.

    In a pluggy plugin manager, the hook implementations registered for each plugin are
    stored in ``_HookCaller`` objects that share the same name as the corresponding hook
    specification; and each ``_HookCaller`` instance is stored under the
    ``plugin_manager.hook`` namespace. For instance:
    ``plugin_manager.hook.name_of_hook_specification``.

    By default, hook implementations are called in last-in-first-out order of registration,
    and pluggy does not provide a built-in way to rearrange the call order of hook
    implementations.

    This function accepts a `_HookCaller` instance and the desired call order of the hook
    implementations (in the form of list of plugin names, or a list of actual ``HookImpl``
    instances in the corresponding hook caller) and reorders the implementations in the hook
    caller accordingly.

    Note: hook implementations are actually stored in *two* separate list attributes in the
    hook caller: ``_HookCaller._wrappers`` and ``_HookCaller._nonwrappers``, according to
    whether the corresponding ``HookImpl`` instance was marked as a wrapper or not.  For
    more, see: https://pluggy.readthedocs.io/en/latest/#wrappers

    Parameters
    ----------
    hook_caller : pluggy.hooks._HookCaller
        The hook caller to reorder
    order : list
        A list of str (plugin names), or ``HookImpl`` instances, in the desired CALL ORDER
        of the hook implementations.  The list does not *need* to include every hook
        implementation in the ``hook_caller.get_hookimpls()``, but those that are not
        included will be moved to the end of the call order.

    Raises
    ------
    TypeError
        If ``order`` is neither entirely strings nor entirely HookImpl
        instances.
    ValueError
        If the 'order' list cannot be interpreted as a list of "plugin_name"
        or ``HookImpl`` instances.
    ValueError
        if 'order' argument has multiple entries for the same implementation.

    Examples
    --------
    Imagine you had a hook specification named ``print_plugin_name``, that expected plugins
    to simply print their own name. An implementation might look like:

    >>> # hook implementation for ``plugin_1``
    >>> @hook_implementation
    ... def print_plugin_name():
    ...     print("plugin_1")

    If three different plugins provided hook implementations. An example call for that hook
    might look like:

    >>> plugin_manager.hook.print_plugin_name()
    plugin_1
    plugin_2
    plugin_3

    If you wanted to rearrange their call order, you could do this:

    >>> hook_caller = plugin_manager.hook.print_plugin_name
    >>> new_order = ["plugin_2", "plugin_3", "plugin_1"]
    >>> permute_hook_implementations(hook_caller, new_order)
    >>> plugin_manager.hook.print_plugin_name()
    plugin_2
    plugin_3
    plugin_1

    You can also just specify one or more item to move them to the front of the call order:
    >>> permute_hook_implementations(hook_caller, ["plugin_3"])
    >>> plugin_manager.hook.print_plugin_name()
    plugin_3
    plugin_2
    plugin_1
    """
    # make sure items in order are unique
    if len(order) != len(set(order)):
        raise ValueError("repeated item in order")

    # get a list of all hook implementations in hook_caller
    hook_implementations = hook_caller.get_hookimpls()
    if len(order) > len(hook_implementations):
        raise ValueError(
            "too many values in order: "
            f"({len(order)} > {len(hook_implementations)})"
        )

    # figure out what list type was entered for ``order`` (or raise error)
    order_type = _guess_order_type(order)

    # build a list of the corresponding type (i.e. a list of HookImpl instances
    # or a list of the names of plugins that implement the current hook spec)
    if order_type == 'plugin_name':
        implementation_list = [imp.plugin_name for imp in hook_implementations]
    else:
        implementation_list = hook_implementations

    # find the current position of each item specified in `order`
    indices = []
    for elem in order:
        try:
            indices.append(implementation_list.index(elem))
        except ValueError as e:
            msg = f"Could not find implementation '{elem}' for {hook_caller}."
            raise ValueError(msg) from e

    # make new lists for the rearranged _wrappers and _nonwrappers
    # see for details on the difference between wrappers and nonwrappers, see:
    # https://pluggy.readthedocs.io/en/latest/#wrappers
    _wrappers: List[HookImpl] = []
    _nonwraps: List[HookImpl] = []
    for i in indices:
        hook_implementation = hook_implementations[i]
        if hook_implementation.hookwrapper:
            # (this hook implementation is a wrapper)
            _wrappers.insert(0, hook_implementation)
        else:
            _nonwraps.insert(0, hook_implementation)

    # remove items that have been pulled, leaving only items that
    # were not specified in `order` argument
    # do this rather than using .pop() above to avoid changing indices
    for i in sorted(indices, reverse=True):
        del hook_implementations[i]

    # if there are any hook_implementations left over, add them to the
    # beginning of their respective lists
    if hook_implementations:
        _wrappers = [
            x for x in hook_implementations if x.hookwrapper
        ] + _wrappers
        _nonwraps = [
            x for x in hook_implementations if not x.hookwrapper
        ] + _nonwraps

    # update the original hook_caller object
    hook_caller._wrappers = _wrappers
    hook_caller._nonwrappers = _nonwraps
