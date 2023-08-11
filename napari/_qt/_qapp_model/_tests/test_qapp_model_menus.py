import pytest

from napari._app_model import constants, get_app
from napari._qt._qapp_model import build_qmodel_menu


@pytest.mark.parametrize('menu_id', list(constants.MenuId))
def test_build_qmodel_menu(make_napari_viewer, qtbot, menu_id):
    """Test that we can build qmenus for all registered menu IDs"""
    app = get_app()

    # Runs setup actions; `init_qactions` and `initialize_plugins`
    make_napari_viewer()
    menu = build_qmodel_menu(menu_id)
    qtbot.addWidget(menu)

    # `>=` because separator bars count as actions
    assert len(menu.actions()) >= len(app.menus.get_menu(menu_id))
