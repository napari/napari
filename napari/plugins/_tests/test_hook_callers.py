import pytest
from pluggy import HookCallError, HookspecMarker, PluginValidationError
from pluggy.hooks import HookImpl

from napari.plugins import _hook_callers, PluginManager
from napari_plugins import HookimplMarker

hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


def MC(methods, kwargs, firstresult=False):
    caller = _hook_callers._multicall
    hookfuncs = []
    for method in methods:
        f = HookImpl(None, "<temp>", method, method.example_impl)
        hookfuncs.append(f)
    return caller(hookfuncs, kwargs, firstresult=firstresult)


def test_call_passing():
    class P1(object):
        @hookimpl
        def m(self, x):
            return 17

    class P2(object):
        @hookimpl
        def m(self, x):
            return 23

    p1 = P1()
    p2 = P2()
    reslist = MC([p1.m, p2.m], {"x": 23})
    assert len(reslist) == 2
    # ensure reversed order
    assert reslist == [23, 17]


def test_keyword_args():
    @hookimpl
    def f(x):
        return x + 1

    class A(object):
        @hookimpl
        def f(self, x, y):
            return x + y

    reslist = MC([f, A().f], dict(x=23, y=24))
    assert reslist == [24 + 23, 24]


def test_keyword_args_with_defaultargs():
    @hookimpl
    def f(x, z=1):
        return x + z

    reslist = MC([f], dict(x=23, y=24))
    assert reslist == [24]


def test_tags_call_error():
    @hookimpl
    def f(x):
        return x

    with pytest.raises(HookCallError):
        MC([f], {})


def test_call_subexecute():
    @hookimpl
    def m():
        return 2

    @hookimpl
    def n():
        return 1

    res = MC([n, m], {}, firstresult=True)
    assert res == 2


def test_call_none_is_no_result():
    @hookimpl
    def m1():
        return 1

    @hookimpl
    def m2():
        return None

    res = MC([m1, m2], {}, firstresult=True)
    assert res == 1
    res = MC([m1, m2], {}, {})
    assert res == [1]


def test_hookwrapper():
    out = []

    @hookimpl(hookwrapper=True)
    def m1():
        out.append("m1 init")
        yield None
        out.append("m1 finish")

    @hookimpl
    def m2():
        out.append("m2")
        return 2

    res = MC([m2, m1], {})
    assert res == [2]
    assert out == ["m1 init", "m2", "m1 finish"]
    out[:] = []
    res = MC([m2, m1], {}, firstresult=True)
    assert res == 2
    assert out == ["m1 init", "m2", "m1 finish"]


def test_hookwrapper_order():
    out = []

    @hookimpl(hookwrapper=True)
    def m1():
        out.append("m1 init")
        yield 1
        out.append("m1 finish")

    @hookimpl(hookwrapper=True)
    def m2():
        out.append("m2 init")
        yield 2
        out.append("m2 finish")

    res = MC([m2, m1], {})
    assert res == []
    assert out == ["m1 init", "m2 init", "m2 finish", "m1 finish"]


def test_hookwrapper_not_yield():
    @hookimpl(hookwrapper=True)
    def m1():
        pass

    with pytest.raises(TypeError):
        MC([m1], {})


def test_hookwrapper_too_many_yield():
    @hookimpl(hookwrapper=True)
    def m1():
        yield 1
        yield 2

    with pytest.raises(RuntimeError) as ex:
        MC([m1], {})
    assert "m1" in str(ex.value)
    assert (__file__ + ":") in str(ex.value)


@pytest.mark.parametrize("exc", [SystemExit])
def test_hookwrapper_exception(exc):
    out = []

    @hookimpl(hookwrapper=True)
    def m1():
        out.append("m1 init")
        yield None
        out.append("m1 finish")

    @hookimpl
    def m2():
        raise exc()

    with pytest.raises(exc):
        MC([m2, m1], {})
    assert out == ["m1 init", "m1 finish"]


@pytest.fixture
def pm():
    """PluginManager fixture that loads some test plugins"""
    pm = PluginManager(project_name='example', autodiscover=False)
    return pm


@pytest.fixture
def hc(pm):
    class Hooks(object):
        @hookspec
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)
    return pm.hook.he_method1


@pytest.fixture
def addmeth(hc):
    def addmeth(tryfirst=False, trylast=False, hookwrapper=False):
        def wrap(func):
            hookimpl(
                tryfirst=tryfirst, trylast=trylast, hookwrapper=hookwrapper
            )(func)
            hc._add_hookimpl(HookImpl(None, "<temp>", func, func.example_impl))
            return func

        return wrap

    return addmeth


def funcs(hookmethods):
    return [hookmethod.function for hookmethod in hookmethods]


def test_adding_nonwrappers(hc, addmeth):
    @addmeth()
    def he_method1():
        pass

    @addmeth()
    def he_method2():
        pass

    @addmeth()
    def he_method3():
        pass

    assert funcs(hc._nonwrappers) == [he_method1, he_method2, he_method3]


def test_adding_nonwrappers_trylast(hc, addmeth):
    @addmeth()
    def he_method1_middle():
        pass

    @addmeth(trylast=True)
    def he_method1():
        pass

    @addmeth()
    def he_method1_b():
        pass

    assert funcs(hc._nonwrappers) == [
        he_method1,
        he_method1_middle,
        he_method1_b,
    ]


def test_adding_nonwrappers_trylast3(hc, addmeth):
    @addmeth()
    def he_method1_a():
        pass

    @addmeth(trylast=True)
    def he_method1_b():
        pass

    @addmeth()
    def he_method1_c():
        pass

    @addmeth(trylast=True)
    def he_method1_d():
        pass

    assert funcs(hc._nonwrappers) == [
        he_method1_d,
        he_method1_b,
        he_method1_a,
        he_method1_c,
    ]


def test_adding_nonwrappers_trylast2(hc, addmeth):
    @addmeth()
    def he_method1_middle():
        pass

    @addmeth()
    def he_method1_b():
        pass

    @addmeth(trylast=True)
    def he_method1():
        pass

    assert funcs(hc._nonwrappers) == [
        he_method1,
        he_method1_middle,
        he_method1_b,
    ]


def test_adding_nonwrappers_tryfirst(hc, addmeth):
    @addmeth(tryfirst=True)
    def he_method1():
        pass

    @addmeth()
    def he_method1_middle():
        pass

    @addmeth()
    def he_method1_b():
        pass

    assert funcs(hc._nonwrappers) == [
        he_method1_middle,
        he_method1_b,
        he_method1,
    ]


def test_adding_wrappers_ordering(hc, addmeth):
    @addmeth(hookwrapper=True)
    def he_method1():
        pass

    @addmeth()
    def he_method1_middle():
        pass

    @addmeth(hookwrapper=True)
    def he_method3():
        pass

    assert funcs(hc._nonwrappers) == [he_method1_middle]
    assert funcs(hc._wrappers) == [he_method1, he_method3]


def test_adding_wrappers_ordering_tryfirst(hc, addmeth):
    @addmeth(hookwrapper=True, tryfirst=True)
    def he_method1():
        pass

    @addmeth(hookwrapper=True)
    def he_method2():
        pass

    assert hc._nonwrappers == []
    assert funcs(hc._wrappers) == [he_method2, he_method1]


def test_hookspec(pm):
    class HookSpec(object):
        @hookspec()
        def he_myhook1(arg1):
            pass

        @hookspec(firstresult=True)
        def he_myhook2(arg1):
            pass

        @hookspec(firstresult=False)
        def he_myhook3(arg1):
            pass

    pm.add_hookspecs(HookSpec)
    assert not pm.hook.he_myhook1.spec.opts["firstresult"]
    assert pm.hook.he_myhook2.spec.opts["firstresult"]
    assert not pm.hook.he_myhook3.spec.opts["firstresult"]


@pytest.mark.parametrize(
    "name", ["hookwrapper", "optionalhook", "tryfirst", "trylast"]
)
@pytest.mark.parametrize("val", [True, False])
def test_hookimpl(name, val):
    @hookimpl(**{name: val})
    def he_myhook1(arg1):
        pass

    if val:
        assert he_myhook1.example_impl.get(name)
    else:
        assert not hasattr(he_myhook1, name)


def test_hookrelay_registry(pm):
    """Verify hook caller instances are registered by name onto the relay
    and can be likewise unregistered."""

    class Api(object):
        @hookspec
        def hello(self, arg):
            "api hook 1"

    pm.add_hookspecs(Api)
    hook = pm.hook
    assert hasattr(hook, "hello")
    assert repr(hook.hello).find("hello") != -1

    class Plugin(object):
        @hookimpl
        def hello(self, arg):
            return arg + 1

    plugin = Plugin()
    pm.register(plugin)
    out = hook.hello(arg=3)
    assert out == [4]
    assert not hasattr(hook, "world")
    pm.unregister(plugin)
    assert hook.hello(arg=3) == []


def test_hookrelay_registration_by_specname(pm):
    """Verify hook caller instances may also be registered by specifying a
    specname option to the hookimpl"""

    class Api(object):
        @hookspec
        def hello(self, arg):
            "api hook 1"

    pm.add_hookspecs(Api)
    hook = pm.hook
    assert hasattr(hook, "hello")
    assert len(pm.hook.hello.get_hookimpls()) == 0

    class Plugin(object):
        @hookimpl(specname="hello")
        def foo(self, arg):
            return arg + 1

    plugin = Plugin()
    pm.register(plugin)
    out = hook.hello(arg=3)
    assert out == [4]


def test_hookrelay_registration_by_specname_raises(pm):
    """Verify using specname still raises the types of errors during registration as it
    would have without using specname."""

    class Api(object):
        @hookspec
        def hello(self, arg):
            "api hook 1"

    pm.add_hookspecs(Api)

    # make sure a bad signature still raises an error when using specname
    class Plugin(object):
        @hookimpl(specname="hello")
        def foo(self, arg, too, many, args):
            return arg + 1

    with pytest.raises(PluginValidationError):
        pm.register(Plugin())

    # make sure check_pending still fails if specname doesn't have a
    # corresponding spec.  EVEN if the function name matches one.
    class Plugin2(object):
        @hookimpl(specname="bar")
        def hello(self, arg):
            return arg + 1

    pm.register(Plugin2())
    with pytest.raises(PluginValidationError):
        pm.check_pending()
