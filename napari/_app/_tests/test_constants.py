from napari._app.constants import _commands


def test_command_titles():
    """make sure all command Ids have a title"""
    for command in _commands.CommandId:
        assert command.title is not None
