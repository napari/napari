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

3. we want to sometimes go directly to calling a specific hook implementation.

NOTE: A version of the code here was submitted to pluggy as PR #253:
https://github.com/pytest-dev/pluggy/pull/253
We *may* want to consider removing some of this code if that merges, but we
will probably just wish to maintain our internal hook looping logic.
"""
import sys
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

from pluggy.callers import HookCallError, _raise_wrapfail, _Result
from pluggy.hooks import HookImpl, _HookCaller as _PluggyHookCaller

from .exceptions import PluginCallError


# Vendored with slight modifications from pluggy.callers._multicall:
# https://github.com/pytest-dev/pluggy/blob/master/src/pluggy/callers.py#L157
def _multicall(
    hook_impls: Sequence[HookImpl],
    caller_kwargs: dict,
    _return_impl: bool = False,
    firstresult: bool = False,
) -> Union[Any, List[Any], Tuple[Any, HookImpl], List[Tuple[Any, HookImpl]]]:
    """Loop through ``hook_impls`` with ``**caller_kwargs`` and return results.

    Parameters
    ----------
    hook_impls : list
        A sequence of hook implementation (HookImpl) objects
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
                # the `hook_impl.enabled` attribute is specific to napari
                # it is not recognized or implemented by pluggy
                if not getattr(hook_impl, 'enabled', True):
                    continue
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
                        err = PluginCallError(
                            hook_impl.plugin_name,
                            hook_impl.plugin.__name__,
                            msg,
                        )
                        err.__cause__ = exc
                        # TODO: storing and retrieving these plugin errors is
                        # being addressed in #1024:
                        # https://github.com/napari/napari/pull/1024

                    if res is not None:
                        results.append(res)
                        if _return_impl:
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

        if _return_impl:
            if firstresult:
                return (
                    outcome.get_result(),
                    impl_hits[0] if impl_hits else None,
                )
            return list(zip(outcome.get_result(), impl_hits))
        return outcome.get_result()


class _HookCaller(_PluggyHookCaller):
    """Adding convenience methods to PluginManager.hook

    In a pluggy plugin manager, the hook implementations registered for each
    plugin are stored in ``_HookCaller`` objects that share the same name as
    the corresponding hook specification; and each ``_HookCaller`` instance is
    stored under the ``plugin_manager.hook`` namespace. For instance:
    ``plugin_manager.hook.name_of_hook_specification``.
    """

    # just for type annotation.  These are the lists that store HookImpls
    _wrappers: List[HookImpl]
    _nonwrappers: List[HookImpl]

    def get_hookimpl_for_plugin(self, plugin_name: str):
        """Return hook implementation instance for ``plugin_name`` if found."""
        try:
            return next(
                imp
                for imp in self.get_hookimpls()
                if imp.plugin_name == plugin_name
            )
        except StopIteration:
            raise KeyError(
                f"No implementation of {self.name} found "
                f"for plugin {plugin_name}."
            )

    def index(self, value: Union[str, HookImpl]) -> int:
        """Return index of plugin_name or a HookImpl in self._nonwrappers"""
        if isinstance(value, HookImpl):
            return self._nonwrappers.index(value)
        elif isinstance(value, str):
            plugin_names = [imp.plugin_name for imp in self._nonwrappers]
            return plugin_names.index(value)
        else:
            raise TypeError(
                "argument provided to index must either be the "
                "(string) name of a plugin, or a HookImpl instance"
            )

    def bring_to_front(self, new_order: Union[List[str], List[HookImpl]]):
        """Move items in ``new_order`` to the front of the call order.

        By default, hook implementations are called in last-in-first-out order
        of registration, and pluggy does not provide a built-in way to
        rearrange the call order of hook implementations.

        This function accepts a `_HookCaller` instance and the desired
        ``new_order`` of the hook implementations (in the form of list of
        plugin names, or a list of actual ``HookImpl`` instances) and reorders
        the implementations in the hook caller accordingly.

        NOTE: hook implementations are actually stored in *two* separate list
        attributes in the hook caller: ``_HookCaller._wrappers`` and
        ``_HookCaller._nonwrappers``, according to whether the corresponding
        ``HookImpl`` instance was marked as a wrapper or not.  This method
        *only* sorts _nonwrappers.
        For more, see: https://pluggy.readthedocs.io/en/latest/#wrappers

        Parameters
        ----------
        new_order :  list of str or list of ``HookImpl`` instances
            The desired CALL ORDER of the hook implementations.  The list
            does *not* need to include every hook implementation in
            ``self.get_hookimpls()``, but those that are not included
            will be left at the end of the call order.

        Raises
        ------
        TypeError
            If any item in ``new_order`` is neither a string (plugin_name) or a
            ``HookImpl`` instance.
        ValueError
            If any item in ``new_order`` is neither the name of a plugin or a
            ``HookImpl`` instance that is present in self._nonwrappers.
        ValueError
            If ``new_order`` argument has multiple entries for the same
            implementation.

        Examples
        --------
        Imagine you had a hook specification named ``print_plugin_name``, that
        expected plugins to simply print their own name. An implementation
        might look like:

        >>> # hook implementation for ``plugin_1``
        >>> @hook_implementation
        ... def print_plugin_name():
        ...     print("plugin_1")

        If three different plugins provided hook implementations. An example
        call for that hook might look like:

        >>> plugin_manager.hook.print_plugin_name()
        plugin_1
        plugin_2
        plugin_3

        If you wanted to rearrange their call order, you could do this:

        >>> new_order = ["plugin_2", "plugin_3", "plugin_1"]
        >>> plugin_manager.hook.print_plugin_name.bring_to_front(new_order)
        >>> plugin_manager.hook.print_plugin_name()
        plugin_2
        plugin_3
        plugin_1

        You can also just specify one or more item to move them to the front
        of the call order:
        >>> plugin_manager.hook.print_plugin_name.bring_to_front(["plugin_3"])
        >>> plugin_manager.hook.print_plugin_name()
        plugin_3
        plugin_2
        plugin_1
        """
        # make sure items in order are unique
        if len(new_order) != len(set(new_order)):
            raise ValueError("repeated item in order")

        # make new lists for the rearranged _nonwrappers
        # for details on the difference between wrappers and nonwrappers, see:
        # https://pluggy.readthedocs.io/en/latest/#wrappers
        _old_nonwrappers = self._nonwrappers.copy()
        _new_nonwrappers: List[HookImpl] = []
        indices = [self.index(elem) for elem in new_order]
        for i in indices:
            _new_nonwrappers.insert(0, _old_nonwrappers[i])

        # remove items that have been pulled, leaving only items that
        # were not specified in ``new_order`` argument
        # do this rather than using .pop() above to avoid changing indices
        for i in sorted(indices, reverse=True):
            del _old_nonwrappers[i]

        # if there are any hook_implementations left over, add them to the
        # beginning of their respective lists (because at call time, these
        # lists are called in reverse order)
        if _old_nonwrappers:
            _new_nonwrappers = [x for x in _old_nonwrappers] + _new_nonwrappers

        # update the _nonwrappers list with the reordered list
        self._nonwrappers = _new_nonwrappers

    def _set_plugin_enabled(self, plugin_name: str, enabled: bool):
        """Enable or disable the hook implementation for a specific plugin.

        Parameters
        ----------
        plugin_name : str
            The name of a plugin implementing ``hook_spec``.
        enabled : bool
            Whether or not the implementation should be enabled.

        Raises
        ------
        KeyError
            If ``plugin_name`` has not provided a hook implementation for this
            hook specification.
        """
        self.get_hookimpl_for_plugin(plugin_name).enabled = enabled

    def enable_plugin(self, plugin_name: str):
        """enable implementation for ``plugin_name``."""
        self._set_plugin_enabled(plugin_name, True)

    def disable_plugin(self, plugin_name: str):
        """disable implementation for ``plugin_name``."""
        self._set_plugin_enabled(plugin_name, False)

    def _call_plugin(self, plugin_name: str, *args, **kwargs):
        implementation = self.get_hookimpl_for_plugin(plugin_name)
        if implementation.hookwrapper:
            raise TypeError("Hook wrappers can not be called directly")

        # pluggy only allows calling hooks with keyword arguments
        if args:
            raise TypeError("hook calling supports only keyword arguments")
        _args: List[Any] = []
        try:
            _args = [kwargs[argname] for argname in implementation.argnames]
        except KeyError:
            for argname in implementation.argnames:
                if argname not in kwargs:
                    raise HookCallError(
                        f"hook call must provide argument {argname}"
                    )

        try:
            return implementation.function(*_args)
        except Exception as exc:
            raise PluginCallError(
                implementation.plugin_name,
                implementation.plugin.__name__,
                msg=(
                    f"Error calling plugin '{implementation.plugin_name}', "
                    f"hook '{str(implementation.function.__name__)}'"
                ),
            ) from exc

    def __call__(
        self,
        *args,
        _skip_impls: Optional[Sequence[HookImpl]] = None,
        _return_impl: bool = False,
        _plugin: Optional[str] = None,
        **kwargs,
    ):
        if args:
            raise TypeError("hook calling supports only keyword arguments")
        assert not self.is_historic()
        if self.spec and self.spec.argnames:
            notincall = (
                set(self.spec.argnames)
                - set(["__multicall__"])
                - set(kwargs.keys())
            )
            if notincall:
                warnings.warn(
                    "Argument(s) {} which are declared in the hookspec "
                    "can not be found in this hook call".format(
                        tuple(notincall)
                    ),
                    stacklevel=2,
                )

        if _plugin:
            return self._call_plugin(_plugin, **kwargs)

        skip_impls = _skip_impls or []
        hookimpls = [
            imp for imp in self.get_hookimpls() if imp not in skip_impls
        ]
        firstresult = self.spec.opts.get("firstresult") if self.spec else False
        return _multicall(
            hookimpls,
            kwargs,
            _return_impl=_return_impl,
            firstresult=firstresult,
        )

        return self._hookexec(self, self.get_hookimpls(), kwargs)
