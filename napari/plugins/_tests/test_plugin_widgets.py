import pytest
from napari_plugin_engine import napari_hook_implementation


def func(x, y):
    pass


def func2(x, y):
    pass


fwidget_args = {
    'single_func': func,
    'list_func': [func, func2],
    'bad_func_tuple': (func, {'call_button': True}),
    'bad_full_func_tuple': (func, {'auto_call': True}, {'area': 'right'}),
    'bad_tuple_list': [(func, {'auto_call': True}), (func2, {})],
    'bad_func': 1,
    'bad_tuple1': (func, 1),
    'bad_tuple2': (func, {}, 1),
    'bad_tuple3': (func, 1, {}),
    'bad_double_tuple': ((func, {}), (func2, {})),
    'bad_magic_kwargs': (func, {"non_magicgui_kwarg": True}),
    'bad_good_magic_kwargs': (func, {'call_button': True, "x": {'max': 200}}),
}


# napari_plugin_manager fixture from napari.conftest
# request, recwarn fixtures are from pytest
@pytest.mark.parametrize('arg', fwidget_args.values(), ids=fwidget_args.keys())
def test_function_widget_registration(
    arg, napari_plugin_manager, request, recwarn
):
    """Test that function widgets get validated and registerd correctly."""

    class Plugin:
        @napari_hook_implementation
        def napari_experimental_provide_function():
            return arg

    napari_plugin_manager.discover_widgets()
    napari_plugin_manager.register(Plugin, name='Plugin')

    f_widgets = napari_plugin_manager._function_widgets

    if 'bad_' in request.node.name:
        assert not f_widgets
        assert len(recwarn) == 1
    else:
        assert f_widgets['Plugin']['func'] == func
        assert len(recwarn) == 0
        if 'list_func' in request.node.name:
            assert f_widgets['Plugin']['func2'] == func2
