"""Modified _hookexec caller.

When using a pluggy PluginManager, when you actually call a hook using their
syntax ``plugin_manager.hook.hook_spec_name(**kwargs)`` it goes through a
relatively complicated chain of methods and aliases before ultimately landing
on the basic function that loops through plugins: ``pluggy.callers._multicall``

This module is here as a patch because:

1. pluggy doesn't indicate WHICH hook implementation returned the result(s)...
   making it hard to provide useful tracebacks and feedback if one of them
   errors.

2. pluggy.callers._multicall does not wrap the actual call to hook
   implementations (``res = hook_impl.function(*args)``) in a try/except,
   meaning that if *any* of the implementations throw an error, the whole hook
   chain fails.

Example
-------
Calling hooks the normal way:
>>> result = plugin_manager.hook.my_hook(arg=arg)

is identical to calling them with execute_hook as follows:
>>> result = execute_hook(plugin_manager.hook.my_hook, arg=arg)

If you want to know which implementation is responsible for each result, use:

>>> # NOTE: this assumes plugin_manager.hook.my_hook has firstresult = True
>>> # otherwise a list would be returned.
>>> (result, imp) = execute_hook(
...                     plugin_manager.hook.my_hook,
...                     arg=arg,
...                     return_impl=True,
...                 )

NOTE: A version code here was submitted to pluggy as PR #253:
https://github.com/pytest-dev/pluggy/pull/253
We may want to consider removing this code if that merges, but we may also just
wish to maintain our internal hook looping logic.
"""
import sys
from typing import Optional, Sequence, Any, List, Tuple, Union

from pluggy.callers import HookCallError, _raise_wrapfail, _Result
from pluggy.hooks import HookImpl, _HookCaller
from .exceptions import PluginError


# Vendored with slight modifications from pluggy.callers._multicall:
# https://github.com/pytest-dev/pluggy/blob/master/src/pluggy/callers.py#L157
def _multicall(
    hook_impls: Sequence[HookImpl],
    caller_kwargs: dict,
    return_impl: bool = False,
    firstresult: bool = False,
) -> Union[Any, List[Any], Tuple[Any, HookImpl], List[Tuple[Any, HookImpl]]]:
    """Loop through ``hook_impls`` with ``**caller_kwargs`` and return results.

    Parameters
    ----------
    hook_impls : list
        A sequence of HookImpl objects
    caller_kwargs : dict
        Keyword:value pairs to pass to each ``hook_impl.function``.  Every
        key in the dict must be present in the ``argnames`` property for each
        ``hook_impl`` in ``hook_impls``.
    return_impl : bool, optional
        If ``True``, results are returned as a 2-tuple of ``(result,
        hook_impl)`` where ``hook_impl`` is the implementation responsible for
        returning the result.
    firstresult : bool, optional
        If ``True``, return the first non-null result found, otherwise, return
        a list of results from all hook implementations, by default False

    Returns
    -------
    Any or Tuple[Any, HookImpl] or List[Any] or List[Tuple[Any, HookImpl]]
        The result(s) retrieved from the hook implementations.
        If ``firstresult` is ``True``, then this function will return the first
        non-None result found when looping through ``hook_impls``.  Otherwise,
        a list of results will be returned.
        If ``return_impl`` is True, then results will be returned as a 2-tuple
        of ``(result, hook_impl)``, where ``hook_impl`` is the implementation
        responsible for returning the result.

    Raises
    ------
    HookCallError
        If one or more of the keys in ``caller_kwargs`` is not present in one
        of the ``hook_impl.argnames``.
    """
    __tracebackhide__ = True
    results = []
    impl_hits = []
    excinfo = None
    try:  # run impl and wrapper setup functions in a loop
        teardowns = []
        try:
            for hook_impl in reversed(hook_impls):
                args: List[Any] = []
                try:
                    args = [
                        caller_kwargs[argname]
                        for argname in hook_impl.argnames
                    ]
                except KeyError:
                    for argname in hook_impl.argnames:
                        if argname not in caller_kwargs:
                            raise HookCallError(
                                "hook call must provide argument %r"
                                % (argname,)
                            )

                if hook_impl.hookwrapper:
                    try:
                        gen = hook_impl.function(*args)
                        next(gen)  # first yield
                        teardowns.append(gen)
                    except StopIteration:
                        _raise_wrapfail(gen, "did not yield")
                else:
                    res = None
                    # this is where the plugin function actually gets called
                    # we put it in a try/except so that if one plugin throws
                    # an exception, we don't lose the whole loop
                    try:
                        res = hook_impl.function(*args)
                    except Exception as exc:
                        msg = (
                            f"Error in plugin '{hook_impl.plugin_name}', "
                            f"hook '{str(hook_impl.function.__name__)}'"
                        )
                        err = PluginError(
                            msg,
                            hook_impl.plugin_name,
                            hook_impl.plugin.__name__,
                        )
                        err.__cause__ = exc
                        # TODO: storing and retrieving these plugin errors is
                        # being addressed in #1024:
                        # https://github.com/napari/napari/pull/1024

                    if res is not None:
                        results.append(res)
                        if return_impl:
                            impl_hits.append(hook_impl)
                        if firstresult:  # halt further impl calls
                            break
        except BaseException:
            excinfo = sys.exc_info()
    finally:
        if firstresult:  # first result hooks return a single value
            outcome = _Result(results[0] if results else None, excinfo)
        else:
            outcome = _Result(results, excinfo)

        # run all wrapper post-yield blocks
        for gen in reversed(teardowns):
            try:
                gen.send(outcome)
                _raise_wrapfail(gen, "has second yield")
            except StopIteration:
                pass

        if return_impl:
            if firstresult:
                return (
                    outcome.get_result(),
                    impl_hits[0] if impl_hits else None,
                )
            return list(zip(outcome.get_result(), impl_hits))
        return outcome.get_result()


def execute_hook(
    hook: _HookCaller,
    skip_impls: Optional[Sequence[HookImpl]] = None,
    return_impl: bool = False,
    **kwargs,
) -> Union[Any, List[Any], Tuple[Any, HookImpl], List[Tuple[Any, HookImpl]]]:
    """Return result(s) from all or some of the implementations for ``hook``.

    Examples
    --------
    Calling hooks the normal way...
    >>> result = plugin_manager.hook.my_hook(arg=arg)

    ...is identical to calling them with this function as follows:
    >>> result = execute_hook(plugin_manager.hook.my_hook, arg=arg)

    The difference is that this function offers the ability to skip certain
    implementations, and also calls our internal :func:`_multicall` function
    above, which wraps each hook implentation call in a try/execpt, and offers
    the option of returning the HookImpl responsible for the result along with
    the result itself.

    Parameters
    ----------
    hook : _HookCaller
        An instance of a pluggy._HookCaller.
    skip_impls : Optional[Sequence[HookImpl]], optional
        Hook implementations to skip when looping, by default None
    return_impl : bool, optional
        If ``True``, results are returned as a 2-tuple of ``(result,
        hook_impl)`` where ``hook_impl`` is the implementation responsible for
        returning the result.
    kwargs : dict
        key/value pairs that will be passed to each hook implementation. These
        should match the ``hook_specification`` for ``hook``.

    Returns
    -------
    Any or Tuple[Any, HookImpl] or List[Any] or List[Tuple[Any, HookImpl]]
        The result(s) retrieved from the hook implementations. See docstring
        of :func:`multicall` for details.
    """
    skip_impls = skip_impls or []
    hookimpls = [imp for imp in hook.get_hookimpls() if imp not in skip_impls]
    firstresult = hook.spec.opts.get("firstresult") if hook.spec else False
    return _multicall(
        hookimpls, kwargs, return_impl=return_impl, firstresult=firstresult
    )
