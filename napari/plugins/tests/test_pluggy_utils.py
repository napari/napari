import pytest

import pluggy
from napari.plugins.manager import permute_hook_implementations

hookimpl = pluggy.HookimplMarker("dummy")


def _get_call_order(hook_caller):
    hookimpls = list(
        reversed([i.plugin_name for i in hook_caller.get_hookimpls()])
    )
    hookimpls.extend([i.plugin_name for i in hook_caller._wrappers])
    return hookimpls


class MySpec(object):
    @pluggy.HookspecMarker("dummy")
    def myhook(self):
        pass


class Plugin_1:
    @hookimpl
    def myhook(self):
        return "p1"


class Plugin_2:
    @hookimpl(tryfirst=True)
    def myhook(self):
        return "p2"


class Plugin_3:
    @hookimpl
    def myhook(self):
        return "p3"


class Plugin_W1:
    @hookimpl(hookwrapper=True)
    def myhook(self):
        yield


class Plugin_W2:
    @hookimpl(hookwrapper=True)
    def myhook(self):
        yield


p1, p2, p3 = Plugin_1(), Plugin_2(), Plugin_3()
w1, w2 = Plugin_W1(), Plugin_W2()


@pytest.fixture
def pm():
    pm = pluggy.PluginManager("dummy")
    pm.add_hookspecs(MySpec)
    pm.register(p1, name='p1')
    pm.register(p2, name='p2')
    pm.register(p3, name='p3')
    pm.register(w1, name='w1')
    pm.register(w2, name='w2')
    return pm


START_ORDER = ['p2', 'p3', 'p1']


@pytest.mark.parametrize(
    'order, expected',
    [
        ([], START_ORDER),
        (['p2'], START_ORDER),
        (['p2', 'p3'], START_ORDER),
        (['p1', 'p2', 'p3'], ['p1', 'p2', 'p3']),
        (['p1', 'p3', 'p2'], ['p1', 'p3', 'p2']),
        (['p1', 'p3'], ['p1', 'p3', 'p2']),
        ([p1, p3], ['p1', 'p3', 'p2']),
        (['p1'], ['p1', 'p2', 'p3']),
        (['p3'], ['p3', 'p2', 'p1']),
        ([p3], ['p3', 'p2', 'p1']),
    ],
)
def test_permute_hook_implementations(pm, order, expected):
    assert pm.hook.myhook() == START_ORDER
    permute_hook_implementations(pm.hook.myhook, order)
    assert pm.hook.myhook() == expected


def test_permute_hook_implementations_raises(pm):
    with pytest.raises(ValueError):
        # this plugin instance is not in the list
        permute_hook_implementations(pm.hook.myhook, [Plugin_W2(), p1])

    with pytest.raises(TypeError):
        # this plugin instance is in the list, but cannot mix types.
        permute_hook_implementations(pm.hook.myhook, [p2, 'p1'])

    with pytest.raises(ValueError):
        # 'p4' is not in the list
        permute_hook_implementations(pm.hook.myhook, ['p1', 'p4'])

    with pytest.raises(ValueError):
        # duplicate entries are not allowed
        permute_hook_implementations(pm.hook.myhook, ['p1', 'p1', 'p2'])

    with pytest.raises(ValueError):
        # too many values
        permute_hook_implementations(
            pm.hook.myhook, ['p1', 'p1', 'p2', 'p4', 'p3', 'p1']
        )
