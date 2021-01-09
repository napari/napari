import pytest
from napari_plugin_engine import napari_hook_implementation

from napari import plugins
from napari.plugins import hook_specifications


def func(x, y):
    pass


def func2(x, y):
    pass


args = {
    'single_func': func,
    'func_tuple': (func, {'call_button': True}),
    'full_func_tuple': (func, {'auto_call': True}, {'area': 'right'}),
    'tuple_list': [(func, {'auto_call': True}, {'area': 'right'})],
    'bad_func': 1,
    'bad_tuple1': (func, 1),
    'bad_tuple2': (func, {}, 1),
    'bad_tuple3': (func, 1, {}),
    'bad_double_tuple': ((func, {}), (func2, {})),
    'bad_magic_kwargs': (func, {"non_magicgui_kwarg": True}),
}


@pytest.mark.parametrize('arg', args.values(), ids=args.keys())
def test_function_widget_registration(
    arg, test_plugin_manager, add_implementation, monkeypatch, request, recwarn
):
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


# def test_dock_widget_registration():
#     pass

#     plugin_manager.hook.napari_experimental_provide_dock_widget.call_historic(
#         result_callback=register_dock_widget, with_impl=True
#     )
