from qtpy.QtCore import QTimer
import pytest
from napari_plugin_engine.manager import temp_path_additions


GOOD_PLUGIN = """
from napari_plugin_engine import HookImplementationMarker

@HookImplementationMarker("test")
def napari_get_reader(path):
    return True
"""


@pytest.fixture
def entrypoint_plugin(tmp_path):
    """An example plugin that uses entry points."""
    (tmp_path / "entrypoint_plugin.py").write_text(GOOD_PLUGIN)
    distinfo = tmp_path / "entrypoint_plugin-1.2.3.dist-info"
    distinfo.mkdir()
    (distinfo / "top_level.txt").write_text('entrypoint_plugin')
    (distinfo / "entry_points.txt").write_text(
        "[app.plugin]\na_plugin = entrypoint_plugin"
    )
    (distinfo / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: a_plugin\n"
        "Version: 1.2.3\n"
        "Author-Email: example@example.com\n"
        "Home-Page: https://www.example.com\n"
        "Requires-Python: >=3.6\n"
    )
    return tmp_path


# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_qt_plugin_list(
    viewer_factory, test_plugin_manager, entrypoint_plugin
):
    """Make sure the plugin list viewer works and has the test plugins."""
    view, viewer = viewer_factory()
    with temp_path_additions(entrypoint_plugin):
        test_plugin_manager.discover(entry_point='app.plugin')
        assert 'a_plugin' in test_plugin_manager.plugins

        def handle_dialog():
            assert hasattr(viewer.window, '_plugin_list')
            table = viewer.window._plugin_list.table
            assert table.rowCount() > 0
            plugins = {
                table.item(i, 0).text() for i in range(table.rowCount())
            }
            assert 'a_plugin' in plugins
            viewer.window._plugin_list.close()

        QTimer.singleShot(100, handle_dialog)
        viewer.window._show_plugin_list(test_plugin_manager)
