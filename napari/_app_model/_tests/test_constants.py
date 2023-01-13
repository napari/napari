from napari._app_model.constants import CommandId, MenuId


def test_command_titles():
    """make sure all command start with napari: and have a title"""
    for command in CommandId:
        assert command.value.startswith('napari:')
        assert command.title is not None


def test_menus():
    """make sure all menus start with napari/"""
    for menu in MenuId:
        assert menu.value.startswith('napari/')
