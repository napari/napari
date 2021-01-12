import pytest
from napari_plugin_engine import napari_hook_implementation

from napari import plugins
from napari.plugins import hook_specifications


def func(x, y):
    pass


def func2(x, y):
    pass


fwidget_args = {
    'single_func': func,
    'func_tuple': (func, {'call_button': True}),
    'full_func_tuple': (func, {'auto_call': True}, {'area': 'right'}),
    'tuple_list': [(func, {'auto_call': True}), (func2, {})],
    'tuple_list2': [(func, {'auto_call': True}), func2],
    'bad_func': 1,
    'bad_tuple1': (func, 1),
    'bad_tuple2': (func, {}, 1),
    'bad_tuple3': (func, 1, {}),
    'bad_double_tuple': ((func, {}), (func2, {})),
    'bad_magic_kwargs': (func, {"non_magicgui_kwarg": True}),
}


# test_plugin_manager and add_implementation fixtures are
#     provided by napari_plugin_engine._testsupport
# monkeypatch, request, recwarn fixtures are from pytest
@pytest.mark.parametrize('arg', fwidget_args.values(), ids=fwidget_args.keys())
def test_function_widget_registration(
    arg, test_plugin_manager, add_implementation, monkeypatch, request, recwarn
):
    """Test that function widgets get validated and registerd correctly."""
    test_plugin_manager.project_name = 'napari'
    test_plugin_manager.add_hookspecs(hook_specifications)
    hook = test_plugin_manager.hook.napari_experimental_provide_function_widget

    with monkeypatch.context() as m:
        registered = {}
        m.setattr(plugins, "function_widgets", registered)

        @napari_hook_implementation
        def napari_experimental_provide_function_widget():
            return arg

        add_implementation(napari_experimental_provide_function_widget)
        hook.call_historic(
            result_callback=plugins.register_function_widget, with_impl=True
        )
        if '[bad_' in request.node.name:
            assert not registered
            assert len(recwarn) == 1
        else:
            assert registered[(None, 'func')][0] == func
            assert len(recwarn) == 0
            if 'tuple_list' in request.node.name:
                assert registered[(None, 'func2')][0] == func2


# test_dock_widget_registration is done in `_qt._tests.test_plugin_widgets`
