import pytest

import pluggy
from napari.plugins.manager import PluginManager

dummy_hook_implementation = pluggy.HookimplMarker("dummy")
dummy_hook_specification = pluggy.HookspecMarker("dummy")


class MySpec:
    @dummy_hook_specification
    def myhook(self):
        pass


class Plugin_1:
    @dummy_hook_implementation
    def myhook(self):
        return "p1"


class Plugin_2:
    @dummy_hook_implementation(tryfirst=True)
    def myhook(self):
        return "p2"


class Plugin_3:
    @dummy_hook_implementation
    def myhook(self):
        return "p3"


class Wrapper:
    @dummy_hook_implementation(hookwrapper=True)
    def myhook(self):
        yield


p1, p2, p3, wrapper = Plugin_1(), Plugin_2(), Plugin_3(), Wrapper()


@pytest.fixture
def dummy_plugin_manager():
    plugin_manager = PluginManager("dummy")
    plugin_manager.add_hookspecs(MySpec)
    plugin_manager.register(p1, name='p1')
    plugin_manager.register(p2, name='p2')
    plugin_manager.register(p3, name='p3')
    plugin_manager.register(wrapper, name='wrapper')
    return plugin_manager


# p2 is first because it was declared with tryfirst=True
# p3 is second because of "last-in-first-out" order
START_ORDER = ['p2', 'p3', 'p1']


@pytest.mark.parametrize(
    'order, expected_result',
    [
        ([], START_ORDER),
        (['p2'], START_ORDER),
        (['p2', 'p3'], START_ORDER),
        (['p1', 'p2', 'p3'], ['p1', 'p2', 'p3']),
        (['p1', 'p3', 'p2'], ['p1', 'p3', 'p2']),
        (['p1', 'p3'], ['p1', 'p3', 'p2']),
        (['p1'], ['p1', 'p2', 'p3']),
        (['p3'], ['p3', 'p2', 'p1']),
    ],
)
def test_reordering_hook_caller(dummy_plugin_manager, order, expected_result):
    """Test that the permute_hook_implementations function reorders hooks."""
    hook_caller = dummy_plugin_manager.hooks.myhook

    assert hook_caller() == START_ORDER
    hook_caller.bring_to_front(order)
    assert hook_caller() == expected_result
    # return to original order
    hook_caller.bring_to_front(START_ORDER)
    assert hook_caller() == START_ORDER

    # try again using HookImpl INSTANCES instead of plugin names
    instances = [hook_caller.get_plugin_implementation(i) for i in order]
    hook_caller.bring_to_front(instances)
    assert hook_caller() == expected_result


def test_reordering_hook_caller_raises(dummy_plugin_manager):
    """Test that invalid calls to permute_hook_implementations raise errors."""
    hook_caller = dummy_plugin_manager.hooks.myhook

    with pytest.raises(TypeError):
        # all items must be the name of a plugin, or a HookImpl instance
        hook_caller.bring_to_front([1, 2])

    with pytest.raises(ValueError):
        # 'wrapper' is the name of a plugin that provides an implementation...
        # but it is a hookwrappers which is not valid for `bring_to_front`
        hook_caller.bring_to_front(['p1', 'wrapper'])

    with pytest.raises(ValueError):
        # 'p4' is not in the list
        hook_caller.bring_to_front(['p1', 'p4'])

    with pytest.raises(ValueError):
        # duplicate entries are not allowed
        hook_caller.bring_to_front(['p1', 'p1', 'p2'])

    with pytest.raises(ValueError):
        # too many values
        hook_caller.bring_to_front(['p1', 'p1', 'p2', 'p4', 'p3', 'p1'])
