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

from pluggy.callers import HookCallError, _raise_wrapfail
from pluggy.hooks import HookImpl
from pluggy.hooks import _HookCaller as _PluggyHookCaller

from ..types import ExcInfo
from .exceptions import PluginCallError


class HookResult:
    """A class to store/modify results from a _multicall hook loop.

    Modified from pluggy.callers._Result.
    Results are accessed in ``.result`` property, which will also raise
    any exceptions that occured during the hook loop.

    Parameters
    ----------
    results : List[Tuple[Any, HookImpl]]
        A list of (result, HookImpl) tuples, with the result and HookImpl
        object responsible for each result collected during a _multicall loop.
    excinfo : tuple
        The output of sys.exc_info() if raised during the multicall loop.
    firstresult : bool, optional
        Whether the hookspec had ``firstresult == True``, by default False.
        If True, self._result, and self.implementation will be single values,
        otherwise they will be lists.
    plugin_errors : list
        A list of any :class:`napari.plugins.exceptions.PluginCallError`
        instances that were created during the multicall loop.

    Attributes
    ----------
    result : list or any
        The result (if ``firstresult``) or results from the hook call.  The
        result property will raise any errors in ``excinfo`` when accessed.
    implementation : list or any
        The HookImpl instance (if ``firstresult``) or instances that were
        responsible for each result in ``result``.
    is_firstresult : bool
        Whether this HookResult came from a ``firstresult`` multicall.
    """

    def __init__(
        self,
        result: List[Tuple[Any, HookImpl]],
        excinfo: Optional[ExcInfo],
        firstresult: bool = False,
        plugin_errors: Optional[List[PluginCallError]] = None,
    ):
        self._result = []
        self.implementation = []
        if result:
            self._result, self.implementation = tuple(zip(*result))
            self._result = list(self._result)
            if firstresult and self._result:
                self._result = self._result[0]
                self.implementation = self.implementation[0]
        self._excinfo = excinfo
        self.is_firstresult = firstresult
        self.plugin_errors = plugin_errors
        # str with name of hookwrapper that override result
        self._modified_by: Optional[str] = None

    @classmethod
    def from_call(cls, func):
        """Used when hookcall monitoring is enabled.

        https://pluggy.readthedocs.io/en/latest/#call-monitoring
        """
        raise NotImplementedError

    def force_result(self, result: Any):
        """Force the result(s) to ``result``.

        This may be used by hookwrappers to alter this result object.

        If the hook was marked as a ``firstresult`` a single value should
        be set otherwise set a (modified) list of results. Any exceptions
        found during invocation will be deleted.
        """
        import inspect

        self._result = result
        self._excinfo = None
        self._modified_by = inspect.stack()[1].function

    @property
    def result(self) -> Union[Any, List[Any]]:
        """Return the result(s) for this hook call.

        If the hook was marked as a ``firstresult`` only a single value
        will be returned otherwise a list of results.
        """
        __tracebackhide__ = True
        if self._excinfo is not None:
            _type, value, traceback = self._excinfo
            if value:
                raise value.with_traceback(traceback)
        return self._result


# Vendored with modifications from pluggy.callers._multicall:
# https://github.com/pytest-dev/pluggy/blob/master/src/pluggy/callers.py#L157
def _multicall(
    hook_impls: Sequence[HookImpl],
    caller_kwargs: dict,
    firstresult: bool = False,
) -> HookResult:
    """Loop through ``hook_impls`` with ``**caller_kwargs`` and return results.

    Parameters
    ----------
    hook_impls : list
        A sequence of hook implementation (HookImpl) objects
    caller_kwargs : dict
        Keyword:value pairs to pass to each ``hook_impl.function``.  Every
        key in the dict must be present in the ``argnames`` property for each
        ``hook_impl`` in ``hook_impls``.
    firstresult : bool, optional
        If ``True``, return the first non-null result found, otherwise, return
        a list of results from all hook implementations, by default False

    Returns
    -------
    outcome : HookResult
        A :class:`HookResult` object that contains the results returned by
        plugins along with other metadata about the call.

    Raises
    ------
    HookCallError
        If one or more of the keys in ``caller_kwargs`` is not present in one
        of the ``hook_impl.argnames``.
    PluginCallError
        If ``firstresult == True`` and a plugin raises an Exception.
    """
    __tracebackhide__ = True
    results = []
    errors: List[PluginCallError] = []
    excinfo: Optional[ExcInfo] = None
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
                        # creating a PluginCallError will store it for later
                        # in plugins.exceptions.PLUGIN_ERRORS
                        errors.append(PluginCallError(hook_impl, cause=exc))
                        # if it was a `firstresult` hook, break and raise now.
                        if firstresult:
                            break

                    if res is not None:
                        results.append((res, hook_impl))
                        if firstresult:  # halt further impl calls
                            break
        except BaseException:
            excinfo = sys.exc_info()
    finally:
        if firstresult and errors:
            raise errors[-1]

        outcome = HookResult(
            results,
            excinfo=excinfo,
            firstresult=firstresult,
            plugin_errors=errors,
        )

        # run all wrapper post-yield blocks
        for gen in reversed(teardowns):
            try:
                gen.send(outcome)
                _raise_wrapfail(gen, "has second yield")
            except StopIteration:
                pass

        return outcome


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

    @property
    def is_firstresult(self):
        return self.spec.opts.get("firstresult") if self.spec else False

    def get_plugin_implementation(self, plugin_name: str):
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
            # inserting because they get called in reverse order.
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
        self.get_plugin_implementation(plugin_name).enabled = enabled

    def enable_plugin(self, plugin_name: str):
        """enable implementation for ``plugin_name``."""
        self._set_plugin_enabled(plugin_name, True)

    def disable_plugin(self, plugin_name: str):
        """disable implementation for ``plugin_name``."""
        self._set_plugin_enabled(plugin_name, False)

    def _call_plugin(self, plugin_name: str, *args, **kwargs):
        """Call the hook implementation for a specific plugin

        Note: this method is not intended to be called directly. Instead, just
        call the instance directly, specifing the ``_plugin`` argument.
        See the ``__call__`` method below.

        Parameters
        ----------
        plugin_name : str
            Name of the plugin

        Returns
        -------
        Any
            Result of implementation call provided by plugin

        Raises
        ------
        TypeError
            If the implementation is a hook wrapper (cannot be called directly)
        TypeError
            If positional arguments are provided
        HookCallError
            If one of the required arguments in the hook specification is not
            present in ``kwargs``.
        PluginCallError
            If an exception is raised when calling the plugin
        """
        self._check_call_kwargs(kwargs)
        implementation = self.get_plugin_implementation(plugin_name)
        if implementation.hookwrapper:
            raise TypeError("Hook wrappers can not be called directly")

        # pluggy only allows calling hooks with keyword arguments
        if args:
            raise TypeError("hook calling supports only keyword arguments")
        _args: List[Any] = []
        # this converts kwargs into positional arguments in the correct order
        # for the hookspec
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
            raise PluginCallError(implementation) from exc

    def call_with_result_obj(
        self, *, _skip_impls: Sequence[HookImpl] = list(), **kwargs
    ) -> HookResult:
        """Call hook implementation(s) for this spec and return HookResult.

        The :class:`HookResult` object carries the result (in its ``result``
        property) but also additional information about the hook call, such
        as the implementation that returned each result and any call errors.

        Parameters
        ----------
        _skip_impls : Sequence[HookImpl], optional
            A list of HookImpl instances that should be *skipped* when calling
            hook implementations, by default None
        **kwargs
            keys should match the names of arguments in the corresponding hook
            specification, values will be passed as arguments to the hook
            implementations.

        Returns
        -------
        result : HookResult
            A :class:`HookResult` object that contains the results returned by
            plugins along with other metadata about the call.

        Raises
        ------
        HookCallError
            If one or more of the keys in ``kwargs`` is not present in
            one of the ``hook_impl.argnames``.
        PluginCallError
            If ``firstresult == True`` and a plugin raises an Exception.
        """
        self._check_call_kwargs(kwargs)
        # the heavy lifting of looping through hook implementations, catching
        # errors and gathering results is handled by the _multicall function.
        return _multicall(
            [imp for imp in self.get_hookimpls() if imp not in _skip_impls],
            kwargs,
            firstresult=self.is_firstresult,
        )

    def __call__(
        self,
        *,
        _plugin: Optional[str] = None,
        _skip_impls: Sequence[HookImpl] = list(),
        **kwargs,
    ) -> Union[Any, List[Any]]:
        """Call hook implementation(s) for this spec and return result(s).

        This is the primary way to call plugin hook implementations.

        Note: Parameters are prefaced by underscores to reduce potential
        conflicts with argument names in hook specifications.  There is a test
        in ``test_hook_specifications.test_annotation_on_hook_specification``
        to ensure that these argument names are never used in one of our
        hookspecs.

        Parameters
        ----------
        _plugin : str, optional
            The name of a specific plugin to use.  By default all
            implementations will be called (though if ``firstresult==True``,
            only the first non-None result will be returned).
        _skip_impls : Sequence[HookImpl], optional
            A list of HookImpl instances that should be *skipped* when calling
            hook implementations, by default None
        **kwargs
            keys should match the names of arguments in the corresponding hook
            specification, values will be passed as arguments to the hook
            implementations.

        Raises
        ------
        HookCallError
            If one or more of the keys in ``kwargs`` is not present in one of
            the ``hook_impl.argnames``.
        PluginCallError
            If ``firstresult == True`` and a plugin raises an Exception.

        Returns
        -------
        result
            If the hookspec was declared with ``firstresult==True``, a single
            result will be returned. Otherwise will return a list of results
            from all hook implementations for this hook caller.

            If ``_plugin`` is provided, will return the single result from the
            specified plugin.
        """
        if _plugin:
            # if a plugin name is specified, just call it directly
            return self._call_plugin(_plugin, **kwargs)

        result = self.call_with_result_obj(_skip_impls=_skip_impls, **kwargs)
        return result.result

    def _check_call_kwargs(self, kwargs):
        """Warn if any keys in the hookspec are not present in this call.

        It's possible to add arguments to hook specifications (as they evolve).
        Here we just emit a warning if there are arguments in the hookspec that
        were not specified in this call, which may mean the call could be
        updated.
        """
        # "historic" hooks can be called with ``call_historic()`` *before*
        # having been registered.  However they must be called with
        # self.call_historic().
        # https://pluggy.readthedocs.io/en/latest/index.html#historic-hooks
        assert (
            not self.is_historic()
        ), 'Historic hooks must be called with `call_historic()`'

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
