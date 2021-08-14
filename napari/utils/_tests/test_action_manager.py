"""
This module test some of the behavior of action manager.
"""
import pytest

from ..action_manager import ActionManager


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
    with pytest.warns(UserWarning):
        assert action_manager.unbind_shortcut('napari:foo_bar') is None


def test_bind_multiple_action(action_manager):
    """
    Test we can have multiple bindings per action
    """

    action_manager.register_action(
        'napari:test_action_2', lambda: None, 'this is a test action', None
    )

    action_manager.bind_shortcut('napari:test_action_2', 'X')
    action_manager.bind_shortcut('napari:test_action_2', 'Y')
    assert action_manager._shortcuts['napari:test_action_2'] == {'X', 'Y'}


def test_bind_unbind_existing_action(action_manager):

    action_manager.register_action(
        'napari:test_action_1', lambda: None, 'this is a test action', None
    )

    assert action_manager.bind_shortcut('napari:test_action_1', 'X') is None
    assert action_manager.unbind_shortcut('napari:test_action_1') == {'X'}
    assert action_manager._shortcuts['napari:test_action_1'] == set()
