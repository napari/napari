from unittest.mock import MagicMock

import pytest

from napari import viewer
from napari._app_model import constants, get_app
from napari._qt._qapp_model import build_qmodel_menu
from napari._qt._qapp_model.qactions import init_qactions
from napari._qt.qt_main_window import Window


@pytest.mark.parametrize('menu_id', list(constants.MenuId))
def test_build_qmodel_menu(qtbot, menu_id):
    """Test that we can build qmenus for all registered menu IDs"""
    app = get_app()

    mock = MagicMock()
    with app.injection_store.register(
        providers={viewer.Viewer: lambda: mock, Window: lambda: mock}
    ):
        init_qactions.cache_clear()
        init_qactions()

        menu = build_qmodel_menu(menu_id)
        qtbot.addWidget(menu)

        # if there is no actions registered for this menu, this is likely a placeholder submenu
        # for example tools/acquisition, these placeholder submenus do not show up under app.menus
        if len(menu.actions()) != 0:
            # `>=` because separator bars count as actions
            assert len(menu.actions()) >= len(app.menus.get_menu(menu_id))
