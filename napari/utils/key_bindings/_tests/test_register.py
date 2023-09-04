from random import shuffle

from app_model.types import KeyBindingRule, KeyCode

from napari.utils.key_bindings.constants import KeyBindingWeights
from napari.utils.key_bindings.register import (
    KeyBindingEntry,
    NapariKeyBindingsRegistry,
)


def test_key_binding_entry_ordering():
    user_block = KeyBindingEntry('', KeyBindingWeights.USER, None)
    user_negate = KeyBindingEntry('-command', KeyBindingWeights.USER, None)
    user_add = KeyBindingEntry('command', KeyBindingWeights.USER, None)

    plugin_block = KeyBindingEntry('', KeyBindingWeights.PLUGIN, None)
    plugin_negate = KeyBindingEntry('-command', KeyBindingWeights.PLUGIN, None)
    plugin_add = KeyBindingEntry('command', KeyBindingWeights.PLUGIN, None)

    core_block = KeyBindingEntry('', KeyBindingWeights.CORE, None)
    core_negate = KeyBindingEntry('-command', KeyBindingWeights.CORE, None)
    core_add = KeyBindingEntry('command', KeyBindingWeights.CORE, None)

    sorted_entries = [
        core_add,
        core_negate,
        core_block,
        plugin_add,
        plugin_negate,
        plugin_block,
        user_add,
        user_negate,
        user_block,
    ]

    unsorted_entries = sorted_entries.copy()
    while unsorted_entries == sorted_entries:
        shuffle(unsorted_entries)

    assert sorted(unsorted_entries) == sorted_entries


def test_registry_insertion():
    registry = NapariKeyBindingsRegistry()

    # ordering by weight
    registry.register_keybinding_rule(
        'a',
        KeyBindingRule(
            primary=KeyCode.KeyA, when=None, weight=KeyBindingWeights.CORE
        ),
    )

    registry.register_keybinding_rule(
        'b',
        KeyBindingRule(
            primary=KeyCode.KeyA, when=None, weight=KeyBindingWeights.USER
        ),
    )

    entries = registry.keymap[KeyCode.KeyA]
    assert [entry.command_id for entry in entries] == ['a', 'b']

    # ordering by weight
    registry.register_keybinding_rule(
        'c',
        KeyBindingRule(
            primary=KeyCode.KeyA, when=None, weight=KeyBindingWeights.PLUGIN
        ),
    )

    assert [entry.command_id for entry in entries] == ['a', 'c', 'b']

    # negate rule
    registry.register_keybinding_rule(
        '-c',
        KeyBindingRule(
            primary=KeyCode.KeyA, when=None, weight=KeyBindingWeights.PLUGIN
        ),
    )

    assert [entry.command_id for entry in entries] == ['a', 'c', '-c', 'b']

    # block rule
    registry.register_keybinding_rule(
        '',
        KeyBindingRule(
            primary=KeyCode.KeyA, when=None, weight=KeyBindingWeights.PLUGIN
        ),
    )

    assert [entry.command_id for entry in entries] == ['a', 'c', '-c', '', 'b']

    # inserts on left side
    registry.register_keybinding_rule(
        'd',
        KeyBindingRule(
            primary=KeyCode.KeyA, when=None, weight=KeyBindingWeights.PLUGIN
        ),
    )

    assert [entry.command_id for entry in entries] == [
        'a',
        'd',
        'c',
        '-c',
        '',
        'b',
    ]
