import numpy as np
from app_model.types import Action

from napari._app_model import get_app
from napari._app_model.constants import MenuId
from napari._app_model.context import LayerListContextKeys as LLCK
from napari.layers import Image

# `builtins` required so there are samples registered, so samples menu exists
# @pytest.mark.parametrize('menu_id', list(constants.MenuId))
# def test_build_qmodel_menu(builtins, make_napari_viewer, qtbot, menu_id):
#     """Test that we can build qmenus for all registered menu IDs."""
#     app = get_app()

#     # Runs setup actions; `init_qactions` and `initialize_plugins`
#     make_napari_viewer()

#     menu = build_qmodel_menu(menu_id)
#     qtbot.addWidget(menu)

#     # `>=` because separator bars count as actions
#     assert len(menu.actions()) >= len(app.menus.get_menu(menu_id))


def test_update_menu_state_context(make_napari_viewer):
    """Test `_update_menu_state` correctly updates enabled/visible state."""
    app = get_app()
    viewer = make_napari_viewer()

    action = Action(
        id='dummy_id',
        title='dummy title',
        callback=lambda: None,
        menus=[{'id': MenuId.MENUBAR_FILE, 'when': (LLCK.num_layers > 0)}],
        enablement=(LLCK.num_layers == 2),
    )
    app.register_action(action)

    dummy_action = viewer.window.file_menu.findAction('dummy_id')

    assert 'dummy_id' in app.commands
    # `dummy_action` should be disabled & not visible, num layers == 0
    assert len(viewer.layers) == 0
    viewer.window._update_menu_state('file_menu')
    assert not dummy_action.isVisible()
    assert not dummy_action.isEnabled()

    # Add a layer, `dummy_action` should be visible but not enabled
    layer_a = Image(np.random.random((10, 10)))
    viewer.layers.append(layer_a)
    assert len(viewer.layers) == 1
    viewer.window._update_menu_state('file_menu')
    assert dummy_action.isVisible()
    assert not dummy_action.isEnabled()

    # Add second layer, `dummy_action` should be enabled and visible
    layer_b = Image(np.random.random((10, 10)))
    viewer.layers.append(layer_b)
    assert len(viewer.layers) == 2
    viewer.window._update_menu_state('file_menu')
    assert dummy_action.isVisible()
    assert dummy_action.isEnabled()
