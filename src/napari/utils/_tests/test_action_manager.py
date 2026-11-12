"""
This module test some of the behavior of action manager.
"""

from unittest.mock import Mock

import pytest

from napari.utils.action_manager import ActionManager
from napari.utils.key_bindings import KeymapProvider
from napari.utils.shortcuts import default_shortcuts


@pytest.fixture
def action_manager():
    """
    Unlike normal napari we use a different instance we have complete control
    over and can throw away this makes it easier.
    """
    return ActionManager()


def test_unbind_non_existing_action(action_manager):
    """
    We test that unbinding an non existing action is ok, this can happen due to
    keybindings in settings while some plugins are deactivated or upgraded.

    We emit a warning but should not fail.
    """
    with pytest.warns(UserWarning, match='Attempting to unbind an action'):
        assert action_manager.unbind_shortcut('napari.foo_bar') is None


def test_bind_multiple_action(action_manager):
    """
    Test we can have multiple bindings per action
    """

    action_manager.register_action(
        'napari.test_action_2', lambda: None, 'this is a test action', None
    )

    action_manager.bind_shortcut('napari.test_action_2', 'X')
    action_manager.bind_shortcut('napari.test_action_2', 'Y')
    assert action_manager._shortcuts['napari.test_action_2'] == ['X', 'Y']


def test_bind_unbind_existing_action(action_manager):
    action_manager.register_action(
        'napari.test_action_1', lambda: None, 'this is a test action', None
    )

    assert action_manager.bind_shortcut('napari.test_action_1', 'X') is None
    assert action_manager.unbind_shortcut('napari.test_action_1') == ['X']
    assert action_manager._shortcuts['napari.test_action_1'] == []


def test_bind_key_generator(action_manager):
    def _sample_generator():
        yield 'X'

    action_manager.register_action(
        'napari.test_action_1',
        _sample_generator,
        'this is a test action',
        None,
    )

    with pytest.raises(ValueError, match='generator functions'):
        action_manager.bind_button('napari.test_action_1', Mock())


@pytest.mark.parametrize('register_name', ['example:run', 'example.run'])
@pytest.mark.parametrize('bind_name', ['example:run', 'example.run'])
def test_legacy_action_names(action_manager, register_name, bind_name):
    class Provider(KeymapProvider):
        pass

    callback = Mock(return_value=42)
    changed = Mock()
    action_manager.events.shortcut_changed.connect(changed)
    action_manager.bind_shortcut(bind_name, 'X')
    action_manager.register_action(register_name, callback, 'Run', Provider)
    action_manager._actions['example.run'].injected = callback

    assert list(action_manager._actions) == ['example.run']
    assert list(action_manager._shortcuts) == ['example.run']
    assert changed.call_args[0][0].name == 'example.run'
    assert Provider.class_keymap
    assert action_manager.trigger(bind_name) == 42

    button = Mock()
    action_manager.bind_button(bind_name, button)
    button.clicked.connect.call_args[0][0]()
    assert callback.call_count == 2
    button.setToolTip.assert_called_once()

    assert action_manager.unbind_shortcut(bind_name) == ['X']
    assert not Provider.class_keymap


@pytest.mark.parametrize('name', ['example.run', 'example.layer.run'])
def test_dotted_action_names(action_manager, name):
    action_manager.register_action(name, lambda: None, 'Run', None)
    assert name in action_manager._actions


@pytest.mark.parametrize('name', ['run', 'example:layer:run'])
def test_invalid_action_names(action_manager, name):
    with pytest.raises(ValueError, match=r'package\.name'):
        action_manager.register_action(name, lambda: None, 'Run', None)


@pytest.mark.parametrize('name', ['napari:', 'napari.'])
def test_bind_placeholder_action(action_manager, name):
    button = Mock()
    action_manager.bind_button(name, button)
    button.clicked.connect.assert_called_once()


def test_default_shortcuts_are_registered():
    from napari.components import ViewerModel
    from napari.utils.action_manager import action_manager

    ViewerModel()
    assert all(':' not in name for name in default_shortcuts)
    assert default_shortcuts.keys() <= action_manager._actions.keys()
