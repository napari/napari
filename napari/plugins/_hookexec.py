"""Modified _hookexec caller.

TODO: TEMPORARY WORKAROUND! remove if/when pluggy merges #253
The code here was submitted to pluggy as a PR:
https://github.com/pytest-dev/pluggy/pull/253

This module is here as a patch because pluggy doesn't indicate WHICH hook
implementation returned the result(s)... making it hard to provide useful
tracebacks and feedback if one of them errors.

Example
-------
In addition to calling hooks the normal way:
>>> plugin_manager.hook.my_hook(arg=arg)

you can use _hookexec from this module as follows:
>>> (result, imp) = _hookexec(
...                     plugin_manager.hook.my_hook,
...                     arg=arg,
...                     with_impl=True,
...                 )
"""
import sys
from typing import Optional, Sequence, Any

from pluggy.callers import HookCallError, _raise_wrapfail, _Result
from pluggy.hooks import HookImpl
from .exceptions import PluginError


# Vendored with slight modifications from pluggy:
# https://github.com/pytest-dev/pluggy/blob/master/src/pluggy/callers.py#L157
def _multicall(hook_impls, caller_kwargs, firstresult=False):
    """Execute a call into multiple python functions/methods and return the
    result(s).

    ``caller_kwargs`` comes from _HookCaller.__call__().
    If ``caller_kwargs`` contains a key ``with_impl`` that evaluates to true,
    results will be returned as 2-tuples of (result, hook_impl) instead of the
    bare result.
    """
    __tracebackhide__ = True
    results = []
    impl_hits = []
    excinfo = None
    with_impl = caller_kwargs.pop("with_impl", False)
    try:  # run impl and wrapper setup functions in a loop
        teardowns = []
        try:
            for hook_impl in reversed(hook_impls):
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
                    # an exception, we don't loose the whole loop
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
                        if with_impl:
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

        if with_impl:
            if firstresult:
                return (
                    outcome.get_result(),
                    impl_hits[0] if impl_hits else None,
                )
            return list(zip(outcome.get_result(), impl_hits))
        return outcome.get_result()


def _hookexec(
    hook, skip_imps: Optional[Sequence[HookImpl]] = None, **kwargs
) -> Any:
    skip_imps = skip_imps or []
    hookimpls = [imp for imp in hook.get_hookimpls() if imp not in skip_imps]
    firstresult = hook.spec.opts.get("firstresult") if hook.spec else False
    return _multicall(hookimpls, kwargs, firstresult=firstresult)
