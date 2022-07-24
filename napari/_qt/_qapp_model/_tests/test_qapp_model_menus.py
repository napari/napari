import pytest

from napari._app_model import app, constants
from napari._qt._qapp_model import build_qmodel_menu


@pytest.mark.parametrize('menu_id', list(constants.MenuId))
def test_build_qmodel_menu(qtbot, menu_id):
    """Test that we can build qmenus for all registered menu IDs"""
    menu = build_qmodel_menu(menu_id)
    qtbot.addWidget(menu)
    assert len(menu.actions()) >= len(app.menus.get_menu(menu_id))
