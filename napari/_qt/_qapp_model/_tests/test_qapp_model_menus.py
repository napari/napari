import pytest

from napari._app_model import constants, get_app
from napari._app_model.constants import CommandId
from napari._qt._qapp_model import build_qmodel_menu


# `builtins` required so there are samples registered, so samples menu exists
@pytest.mark.parametrize('menu_id', list(constants.MenuId))
def test_build_qmodel_menu(builtins, make_napari_viewer, qtbot, menu_id):
    """Test that we can build qmenus for all registered menu IDs."""
    app = get_app()

    # Runs setup actions; `init_qactions` and `initialize_plugins`
    make_napari_viewer()

    menu = build_qmodel_menu(menu_id)
    qtbot.addWidget(menu)

    # `>=` because separator bars count as actions
    assert len(menu.actions()) >= len(app.menus.get_menu(menu_id))


def test_update_enabled_context(make_napari_viewer, builtins):
    """Test that `Window._update_enabled` correctly updates menu state."""
    app = get_app()
    viewer = make_napari_viewer()

    save_layers_action = viewer.window.file_menu.findAction(
        CommandId.DLG_SAVE_LAYERS
    )
    # Check 'Save All Layers...' is not enabled when number of layers is 0
    assert len(viewer.layers) == 0
    viewer.window._update_enabled('file_menu')
    assert not save_layers_action.isEnabled()

    # Add a layer and check 'Save All Layers...' is enabled
    app.commands.execute_command('napari.astronaut')
    assert len(viewer.layers) == 1
    viewer.window._update_enabled('file_menu')
    assert save_layers_action.isEnabled()
