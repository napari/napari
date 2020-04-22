import pytest

from naplugi import (
    HookCallError,
    HookImpl,
    HookimplMarker,
    HookspecMarker,
    PluginManager,
    PluginValidationError,
    callers,
)

example_hookspec = HookspecMarker("example")
example_implementation = HookimplMarker("example")


def multicall(methods, kwargs, firstresult=False):
    """utility function to execute the hook implementations loop"""
    caller = callers._multicall
    hookfuncs = []
    for method in methods:
        f = HookImpl(method, **method.example_impl)
        hookfuncs.append(f)
    # our _multicall function returns our own HookResult object.
    # so to make these pluggy tests pass, we have to access .result to mimic
    # the old behavior (that directly returns results).
    return caller(hookfuncs, kwargs, firstresult=firstresult).result


def test_multicall_passing():
    class Plugin1:
        @example_implementation
        def method(self, x):
            return 17

    class Plugin2:
        @example_implementation
        def method(self, x):
            return 23

    p1 = Plugin1()
    p2 = Plugin2()
    result_list = multicall([p1.method, p2.method], {"x": 23})
    assert len(result_list) == 2
    # ensure reversed order
    assert result_list == [23, 17]


def test_keyword_args():
    @example_implementation
    def func(x):
        return x + 1

    class Plugin:
        @example_implementation
        def func(self, x, y):
            return x + y

    reslist = multicall([func, Plugin().func], {"x": 23, "y": 24})
    assert reslist == [24 + 23, 24]


def test_keyword_args_with_defaultargs():
    @example_implementation
    def func(x, z=1):
        return x + z

    reslist = multicall([func], {"x": 23, "y": 24})
    assert reslist == [24]


def test_tags_call_error():
    @example_implementation
    def func(x):
        return x

    with pytest.raises(HookCallError):
        multicall([func], {})


def test_call_subexecute():
    @example_implementation
    def func1():
        return 2

    @example_implementation
    def func2():
        return 1

    assert multicall([func2, func1], {}, firstresult=True) == 2


def test_call_none_is_no_result():
    @example_implementation
    def func1():
        return 1

    @example_implementation
    def func2():
        return None

    assert multicall([func1, func2], {}, firstresult=True) == 1
    assert multicall([func1, func2], {}, {}) == [1]


def test_hookwrapper():
    out = []

    @example_implementation(hookwrapper=True)
    def func1():
        out.append("func1 init")
        yield None
        out.append("func1 finish")

    @example_implementation
    def func2():
        out.append("func2")
        return 2

    assert multicall([func2, func1], {}) == [2]
    assert out == ["func1 init", "func2", "func1 finish"]
    out = []
    assert multicall([func2, func1], {}, firstresult=True) == 2
    assert out == ["func1 init", "func2", "func1 finish"]


def test_hookwrapper_order():
    out = []

    @example_implementation(hookwrapper=True)
    def func1():
        out.append("func1 init")
        yield 1
        out.append("func1 finish")

    @example_implementation(hookwrapper=True)
    def func2():
        out.append("func2 init")
        yield 2
        out.append("func2 finish")

    assert multicall([func2, func1], {}) == []
    assert out == ["func1 init", "func2 init", "func2 finish", "func1 finish"]


def test_hookwrapper_not_yield():
    @example_implementation(hookwrapper=True)
    def func1():
        pass

    with pytest.raises(TypeError):
        multicall([func1], {})


def test_hookwrapper_too_many_yield():
    @example_implementation(hookwrapper=True)
    def func1():
        yield 1
        yield 2

    with pytest.raises(RuntimeError) as ex:
        multicall([func1], {})
    assert "func1" in str(ex.value)
    assert (__file__ + ":") in str(ex.value)


@pytest.mark.parametrize("exc", [SystemExit])
def test_hookwrapper_exception(exc):
    out = []

    @example_implementation(hookwrapper=True)
    def func1():
        out.append("func1 init")
        yield None
        out.append("func1 finish")

    @example_implementation
    def func2():
        raise exc()

    with pytest.raises(exc):
        multicall([func2, func1], {})
    assert out == ["func1 init", "func1 finish"]


@pytest.fixture
def example_plugin_manager():
    """PluginManager fixture that loads some test plugins"""
    example_plugin_manager = PluginManager(
        project_name='example', autodiscover=False
    )
    return example_plugin_manager


@pytest.fixture
def hook_caller(example_plugin_manager):
    class Hooks:
        @example_hookspec
        def method1(self, arg):
            pass

    example_plugin_manager.add_hookspecs(Hooks)
    return example_plugin_manager.hook.method1


@pytest.fixture
def addmeth(hook_caller):
    def addmeth(tryfirst=False, trylast=False, hookwrapper=False):
        def wrap(func):
            example_implementation(
                tryfirst=tryfirst, trylast=trylast, hookwrapper=hookwrapper
            )(func)
            hook_caller._add_hookimpl(HookImpl(func, **func.example_impl))
            return func

        return wrap

    return addmeth


def funcs(hookmethods):
    return [hookmethod.function for hookmethod in hookmethods]


def test_adding_nonwrappers(hook_caller, addmeth):
    @addmeth()
    def method1():
        pass

    @addmeth()
    def method2():
        pass

    @addmeth()
    def method3():
        pass

    assert funcs(hook_caller._nonwrappers) == [method1, method2, method3]


def test_adding_nonwrappers_trylast(hook_caller, addmeth):
    @addmeth()
    def method1_middle():
        pass

    @addmeth(trylast=True)
    def method1():
        pass

    @addmeth()
    def method1_b():
        pass

    assert funcs(hook_caller._nonwrappers) == [
        method1,
        method1_middle,
        method1_b,
    ]


def test_adding_nonwrappers_trylast3(hook_caller, addmeth):
    @addmeth()
    def method1_a():
        pass

    @addmeth(trylast=True)
    def method1_b():
        pass

    @addmeth()
    def method1_c():
        pass

    @addmeth(trylast=True)
    def method1_d():
        pass

    assert funcs(hook_caller._nonwrappers) == [
        method1_d,
        method1_b,
        method1_a,
        method1_c,
    ]


def test_adding_nonwrappers_trylast2(hook_caller, addmeth):
    @addmeth()
    def method1_middle():
        pass

    @addmeth()
    def method1_b():
        pass

    @addmeth(trylast=True)
    def method1():
        pass

    assert funcs(hook_caller._nonwrappers) == [
        method1,
        method1_middle,
        method1_b,
    ]


def test_adding_nonwrappers_tryfirst(hook_caller, addmeth):
    @addmeth(tryfirst=True)
    def method1():
        pass

    @addmeth()
    def method1_middle():
        pass

    @addmeth()
    def method1_b():
        pass

    assert funcs(hook_caller._nonwrappers) == [
        method1_middle,
        method1_b,
        method1,
    ]


def test_adding_wrappers_ordering(hook_caller, addmeth):
    @addmeth(hookwrapper=True)
    def method1():
        pass

    @addmeth()
    def method1_middle():
        pass

    @addmeth(hookwrapper=True)
    def method3():
        pass

    assert funcs(hook_caller._nonwrappers) == [method1_middle]
    assert funcs(hook_caller._wrappers) == [method1, method3]


def test_adding_wrappers_ordering_tryfirst(hook_caller, addmeth):
    @addmeth(hookwrapper=True, tryfirst=True)
    def method1():
        pass

    @addmeth(hookwrapper=True)
    def method2():
        pass

    assert hook_caller._nonwrappers == []
    assert funcs(hook_caller._wrappers) == [method2, method1]


def test_hookspec(example_plugin_manager):
    class HookSpec:
        @example_hookspec()
        def he_myhook1(arg1):
            pass

        @example_hookspec(firstresult=True)
        def he_myhook2(arg1):
            pass

        @example_hookspec(firstresult=False)
        def he_myhook3(arg1):
            pass

    example_plugin_manager.add_hookspecs(HookSpec)
    assert not example_plugin_manager.hook.he_myhook1.spec.opts["firstresult"]
    assert example_plugin_manager.hook.he_myhook2.spec.opts["firstresult"]
    assert not example_plugin_manager.hook.he_myhook3.spec.opts["firstresult"]


@pytest.mark.parametrize(
    "name", ["hookwrapper", "optionalhook", "tryfirst", "trylast"]
)
@pytest.mark.parametrize("val", [True, False])
def test_hookimpl(name, val):
    @example_implementation(**{name: val})
    def he_myhook1(arg1):
        pass

    if val:
        assert he_myhook1.example_impl.get(name)
    else:
        assert not hasattr(he_myhook1, name)


def test_hookrelay_registry(example_plugin_manager):
    """Verify hook caller instances are registered by name onto the relay
    and can be likewise unregistered."""

    class Api:
        @example_hookspec
        def hello(self, arg):
            "api hook 1"

    example_plugin_manager.add_hookspecs(Api)
    hook = example_plugin_manager.hook
    assert hasattr(hook, "hello")
    assert repr(hook.hello).find("hello") != -1

    class Plugin:
        @example_implementation
        def hello(self, arg):
            return arg + 1

    plugin = Plugin()
    example_plugin_manager.register(plugin)
    out = hook.hello(arg=3)
    assert out == [4]
    assert not hasattr(hook, "world")
    example_plugin_manager.unregister(module=plugin)
    assert hook.hello(arg=3) == []


def test_hookrelay_registration_by_specname(example_plugin_manager):
    """Verify hook caller instances may also be registered by specifying a
    specname option to the hookimpl"""

    class Api:
        @example_hookspec
        def hello(self, arg):
            "api hook 1"

    example_plugin_manager.add_hookspecs(Api)
    hook = example_plugin_manager.hook
    assert hasattr(hook, "hello")
    assert len(example_plugin_manager.hook.hello.get_hookimpls()) == 0

    class Plugin:
        @example_implementation(specname="hello")
        def foo(self, arg):
            return arg + 1

    plugin = Plugin()
    example_plugin_manager.register(plugin)
    out = hook.hello(arg=3)
    assert out == [4]


def test_hookrelay_registration_by_specname_raises(example_plugin_manager):
    """Verify using specname still raises the types of errors during registration as it
    would have without using specname."""

    class Api:
        @example_hookspec
        def hello(self, arg):
            "api hook 1"

    example_plugin_manager.add_hookspecs(Api)

    # make sure a bad signature still raises an error when using specname
    class Plugin:
        @example_implementation(specname="hello")
        def foo(self, arg, too, many, args):
            return arg + 1

    with pytest.raises(PluginValidationError):
        example_plugin_manager.register(Plugin())

    # make sure check_pending still fails if specname doesn't have a
    # corresponding spec.  EVEN if the function name matches one.
    class Plugin2:
        @example_implementation(specname="bar")
        def hello(self, arg):
            return arg + 1

    example_plugin_manager.register(Plugin2())
    with pytest.raises(PluginValidationError):
        example_plugin_manager.check_pending()
