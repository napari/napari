from typing import Callable, Optional
from unittest.mock import Mock, patch

import pytest

from napari.utils._injection import set_providers
from napari.utils.actions import (
    Action,
    CommandsRegistry,
    KeybindingsRegistry,
    MenuId,
    MenuRegistry,
    register_action,
)
from napari.utils.actions._types import CommandId
from napari.utils.context import LayerListContextKeys

PRIMARY_KEY = 'ctrl+a'
OS_KEY = 'ctrl+b'

KWARGS = [
    {},
    dict(enablement=LayerListContextKeys.active_layer_is_rgb),
    dict(menus=[{'id': MenuId.LAYERLIST_CONTEXT}]),
    dict(enablement='3 >= 1', menus=[{'id': MenuId.LAYERLIST_CONTEXT}]),
    dict(keybindings=[{'primary': PRIMARY_KEY}]),
    dict(
        keybindings=[
            {
                'primary': PRIMARY_KEY,
                'mac': OS_KEY,
                'windows': OS_KEY,
                'linux': OS_KEY,
            }
        ]
    ),
    dict(
        keybindings=[{'primary': 'ctrl+a'}],
        menus=[{'id': MenuId.LAYERLIST_CONTEXT}],
    ),
]


@pytest.fixture
def cmd_reg():
    reg = CommandsRegistry()
    reg.registered_emit = Mock()  # type: ignore
    reg.registered.connect(reg.registered_emit)  # type: ignore
    with patch.object(CommandsRegistry, 'instance', return_value=reg):
        yield reg
    reg._commands.clear()


@pytest.fixture
def key_reg():
    reg = KeybindingsRegistry()
    reg.registered_emit = Mock()  # type: ignore
    reg.registered.connect(reg.registered_emit)  # type: ignore
    with patch.object(KeybindingsRegistry, 'instance', return_value=reg):
        yield reg
    reg._coreKeybindings.clear()


@pytest.fixture
def menu_reg():
    reg = MenuRegistry()
    reg.menus_changed_emit = Mock()  # type: ignore
    reg.menus_changed.connect(reg.menus_changed_emit)  # type: ignore
    with patch.object(MenuRegistry, 'instance', return_value=reg):
        yield reg
    reg._menu_items.clear()


@pytest.mark.parametrize('kwargs', KWARGS)
@pytest.mark.parametrize('mode', ['str', 'decorator', 'action'])
def test_register_action_decorator(
    kwargs,
    cmd_reg: CommandsRegistry,
    key_reg: KeybindingsRegistry,
    menu_reg: MenuRegistry,
    mode,
):
    # make sure mocks are working
    assert not list(cmd_reg)
    assert not list(key_reg)
    assert not list(menu_reg)

    dispose: Optional[Callable] = None
    cmd_id = CommandId('cmd.id')
    kwargs['title'] = 'Test title'

    # register the action
    if mode == 'decorator':

        @register_action(cmd_id, **kwargs)
        def f1(x: int):
            return x

        assert f1(1) == 1  # decorator returns the function

    else:

        def f2(x: int):
            return x

        if mode == 'str':
            dispose = register_action(cmd_id, run=f2, **kwargs)

        elif mode == 'action':
            action = Action(id=cmd_id, run=f2, **kwargs)
            dispose = register_action(action)

    # make sure the command is registered
    assert cmd_id in cmd_reg
    assert list(cmd_reg)
    # make sure an event was emitted signaling the command was registered
    cmd_reg.registered_emit.assert_called_once_with(cmd_id)  # type: ignore

    # make sure we can call the command, and that we can inject dependencies.
    with set_providers({int: lambda: 2}):
        assert cmd_reg.execute_command(cmd_id).result() == 2

    # make sure menus are registered if specified
    if menus := kwargs.get('menus'):
        for entry in menus:
            assert entry['id'] in menu_reg
            menu_reg.menus_changed_emit.assert_called_with({entry['id']})  # type: ignore
    else:
        assert not list(menu_reg)

    # make sure keybindings are registered if specified
    if keybindings := kwargs.get('keybindings'):
        for entry in keybindings:
            key = PRIMARY_KEY if len(entry) == 1 else OS_KEY  # see KWARGS[5]
            assert any(i.keybinding == key for i in key_reg)
            key_reg.registered_emit.assert_called()  # type: ignore
    else:
        assert not list(key_reg)

    # if we're not using the decorator, check that calling the dispose
    # function removes everything.  (the decorator returns the function, so can't
    # return the dispose function)
    if dispose:
        dispose()
        assert not list(cmd_reg)
        assert not list(key_reg)
        assert not list(menu_reg)


def test_errors():
    with pytest.raises(ValueError, match="'title' is required"):
        register_action('cmd_id')  # type: ignore
    with pytest.raises(TypeError, match="must be a string or an Action"):
        register_action(None)  # type: ignore


def test_instances():
    assert isinstance(MenuRegistry().instance(), MenuRegistry)
    assert isinstance(
        KeybindingsRegistry().instance(),
        KeybindingsRegistry,
    )
    assert isinstance(CommandsRegistry().instance(), CommandsRegistry)
