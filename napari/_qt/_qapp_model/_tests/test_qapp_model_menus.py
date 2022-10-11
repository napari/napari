from unittest.mock import MagicMock

import pytest

from napari import Viewer
from napari._app_model import constants, get_app
from napari._qt._qapp_model import build_qmodel_menu


@pytest.mark.parametrize('menu_id', list(constants.MenuId))
def test_build_qmodel_menu(qtbot, menu_id):
    """Test that we can build qmenus for all registered menu IDs"""
    app = get_app()

    mock = MagicMock()
    with app.injection_store.register(providers={Viewer: lambda: mock}):
        print(
            f'App injection store providers:\n{app._injection_store._providers}'
        )
        menu = build_qmodel_menu(menu_id)
        qtbot.addWidget(menu)
        assert len(menu.actions()) >= len(app.menus.get_menu(menu_id))
