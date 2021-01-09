import pytest
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget

from napari import Viewer, plugins
from napari.plugins import hook_specifications


class Widg1(QWidget):
    pass


class Widg2(QWidget):
    def __init__(self, napari_viewer):
        pass


class Widg3(QWidget):
    def __init__(self, v: Viewer):
        pass


dwidget_args = {
    'single_class': Widg1,
    'class_tuple': (Widg1, {'area': 'right'}),
    'tuple_list': [(Widg1, {'area': 'right'}), (Widg2, {})],
    'tuple_list2': [(Widg1, {'area': 'right'}), Widg2],
    'bad_class': 1,
    'bad_tuple1': (Widg1, 1),
    'bad_double_tuple': ((Widg1, {}), (Widg2, {})),
}


@pytest.mark.parametrize('arg', dwidget_args.values(), ids=dwidget_args.keys())
def test_dock_widget_registration(
    arg, test_plugin_manager, add_implementation, monkeypatch, request, recwarn
):
    test_plugin_manager.project_name = 'napari'
    test_plugin_manager.add_hookspecs(hook_specifications)
    hook = test_plugin_manager.hook.napari_experimental_provide_dock_widget

    with monkeypatch.context() as m:
        registered = {}
        m.setattr(plugins, "dock_widgets", registered)

        @napari_hook_implementation
        def napari_experimental_provide_dock_widget():
            return arg

        add_implementation(napari_experimental_provide_dock_widget)
        hook.call_historic(
            result_callback=plugins.register_dock_widget, with_impl=True
        )
        if '[bad_' in request.node.name:
            assert len(recwarn) == 1
            assert not registered
        else:
            assert len(recwarn) == 0
            assert registered[(None, 'Widg1')][0] == Widg1
            if 'tuple_list' in request.node.name:
                assert registered[(None, 'Widg2')][0] == Widg2
