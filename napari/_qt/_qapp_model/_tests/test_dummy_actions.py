from napari._app_model.constants._menus import MenuId
from napari._app_model.utils import to_id_key


def assert_empty_keys_in_context(context):
    for menu_id in MenuId.contributables():
        context_key = f'{to_id_key(menu_id)}_empty'
        assert context_key in context


def test_menu_viewer_relaunch(make_napari_viewer):
    viewer = make_napari_viewer()
    ll_context = (
        viewer.window._qt_viewer._layers.model().sourceModel()._root._ctx
    )
    assert_empty_keys_in_context(ll_context)
    viewer.close()

    viewer2 = make_napari_viewer()
    ll_context = (
        viewer2.window._qt_viewer._layers.model().sourceModel()._root._ctx
    )
    # prior to #7106, this would fail
    assert_empty_keys_in_context(ll_context)
    viewer2.close()

    # prior to #7106, creating this viewer would error
    make_napari_viewer()
