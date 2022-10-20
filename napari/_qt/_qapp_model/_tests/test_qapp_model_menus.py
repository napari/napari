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
        init_qactions()
        menu = build_qmodel_menu(menu_id)
        qtbot.addWidget(menu)

        # empty menu not registering actions, for example help menu
        if not menu.actions():
            return

        # `>=` because separator bars count as actions
        assert len(menu.actions()) >= len(app.menus.get_menu(menu_id))
