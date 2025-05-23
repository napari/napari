from napari._app_model.constants import MenuId


def test_menus():
    """make sure all menus start with napari/"""
    for menu in MenuId:
        assert menu.value.startswith('napari/')
