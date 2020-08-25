import pytest
from napari_plugin_engine.manager import temp_path_additions

from napari._qt.dialogs.qt_plugin_table import QtPluginTable

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
def test_qt_plugin_list(test_plugin_manager, entrypoint_plugin, qtbot):
    """Make sure the plugin list viewer works and has the test plugins."""

    with temp_path_additions(entrypoint_plugin):
        test_plugin_manager.discover(entry_point='app.plugin')
        assert 'a_plugin' in test_plugin_manager.plugins
        dialog = QtPluginTable(None, test_plugin_manager)
        qtbot.addWidget(dialog)
        assert dialog.table.rowCount() > 0
        plugins = {
            dialog.table.item(i, 0).text()
            for i in range(dialog.table.rowCount())
        }
        assert 'a_plugin' in plugins


def test_dialog_create(qtbot):
    dialog = QtPluginTable(None)
    qtbot.addWidget(dialog)
    assert dialog.table.rowCount() >= 2
    assert dialog.table.columnCount() == 6
