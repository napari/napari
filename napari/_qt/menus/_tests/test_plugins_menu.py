from npe2 import DynamicPlugin
from qtpy.QtWidgets import QWidget


class DummyWidget(QWidget):
    pass


def test_plugin_display_name_use_for_multiple_widgets(
    make_napari_viewer, tmp_plugin: DynamicPlugin
):
    """For plugin with more than two widgets, should use plugin_display for building the menu"""

    @tmp_plugin.contribute.widget(display_name='Widget 1')
    def widget1():
        return DummyWidget()

    @tmp_plugin.contribute.widget(display_name='Widget 2')
    def widget2():
        return DummyWidget()

    assert tmp_plugin.display_name == 'Temp Plugin'
    viewer = make_napari_viewer()
    # the submenu should use the `display_name` from manifest
    plugin_action_menu = viewer.window.plugins_menu.actions()[3].menu()
    assert plugin_action_menu.title() == tmp_plugin.display_name
    # Now ensure that the actions are still correct
    # trigger the action, opening the first widget: `Widget 1`
    assert len(viewer.window._dock_widgets) == 0
    plugin_action_menu.actions()[0].trigger()
    assert len(viewer.window._dock_widgets) == 1
    assert list(viewer.window._dock_widgets.data)[0] == 'Widget 1 (tmp_plugin)'


def test_plugin_manager(make_napari_viewer):
    """Test that the plugin manager is accessible from the viewer"""
    viewer = make_napari_viewer()

    plugins_menu = viewer.window.plugins_menu
    assert plugins_menu._plugin_manager_dialog_cls is not None

    actions = plugins_menu.actions()
    for action in actions:
        if action.text() == "Plugin Manager":
            break
    else:
        found = [action.text() for action in actions]
        raise AssertionError(
            f'Plugin Manager menu item not found. Only found: {", ".join(found)}'
        )
