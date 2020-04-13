from qtpy.QtCore import QTimer
import os


def test_qt_plugin_list(viewer_factory):
    """Make sure the plugin list viewer works and has the test plugins."""
    view, viewer = viewer_factory()

    from napari.plugins import _tests, get_plugin_manager

    fixture_path = os.path.join(os.path.dirname(_tests.__file__), 'fixtures')
    plugin_manager = get_plugin_manager()
    plugin_manager.discover(fixture_path)

    def handle_dialog():
        assert hasattr(viewer.window, '_plugin_list')
        table = viewer.window._plugin_list.table
        assert table.rowCount() > 0
        plugins = {table.item(i, 0).text() for i in range(table.rowCount())}
        assert 'working' in plugins
        viewer.window._plugin_list.close()

    QTimer.singleShot(100, handle_dialog)
    viewer.window._show_plugin_list()
