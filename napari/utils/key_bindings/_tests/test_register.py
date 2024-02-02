from random import shuffle

from app_model.expressions import parse_expression
from app_model.types import KeyBinding, KeyBindingRule, KeyCode, KeyMod

from napari.utils.key_bindings.constants import KeyBindingWeights
from napari.utils.key_bindings.register import (
    KeyBindingEntry,
    NapariKeyBindingsRegistry,
    filter_entries_by_command,
    group_entries_by_when,
)


def test_filter_entries_by_command():
    block = KeyBindingEntry('', 0, None)
    negate = KeyBindingEntry('-command_a', 0, None)
    add = KeyBindingEntry('command_a', 0, None)

    entries = [
        block,
        KeyBindingEntry('command_b', 0, None),
        negate,
        KeyBindingEntry('-command_b', 0, None),
        add,
    ]

    assert list(filter_entries_by_command(entries, 'command_a')) == [
        negate,
        add,
    ]


def test_group_entries_by_when():
    a = KeyBindingEntry('command_a', 0, parse_expression('pigs_fly'))
    b = KeyBindingEntry('-command_a', 0, parse_expression('pigs_fly'))
    c = KeyBindingEntry('', 0, parse_expression('answer == 42'))
    d = KeyBindingEntry('command_b', 0, parse_expression('answer == 42'))
    e = KeyBindingEntry('command_c', 0, None)

    groups = group_entries_by_when([a, b, c, d, e])

    assert groups[None] == [e]
    assert groups['pigs_fly'] == [a, b]
    assert groups['answer == 42'] == [c, d]


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


def test_registry_get_non_canceling_entries():
    registry = NapariKeyBindingsRegistry()

    # CORE ENTRIES
    registry.register_keybinding_rule(
        'select_all',
        KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.CORE,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        'undo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyCode.KeyZ,
            weight=KeyBindingWeights.CORE,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        'redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ,
            weight=KeyBindingWeights.CORE,
            when=parse_expression('~alternative_redo'),
        ),
    )

    registry.register_keybinding_rule(
        'redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyCode.KeyY,
            weight=KeyBindingWeights.CORE,
            when=parse_expression('alternative_redo'),
        ),
    )

    entries = registry.get_non_canceling_entries('select_all')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(
        KeyMod.Shift | KeyCode.KeyA
    )

    entries = registry.get_non_canceling_entries('undo')
    assert len(entries) == 1
    assert entries[0].command_id == 'undo'

    entries = registry.get_non_canceling_entries('redo')
    assert len(entries) == 2
    assert {entry.keybinding for entry in entries} == {
        KeyBinding.from_int(KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ),
        KeyBinding.from_int(KeyMod.CtrlCmd | KeyCode.KeyY),
    }

    # PLUGIN ENTRIES
    registry.register_keybinding_rule(
        '-select_all',
        KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.PLUGIN,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        'select_all',
        KeyBindingRule(
            primary=KeyCode.KeyA,
            weight=KeyBindingWeights.PLUGIN,
            when=parse_expression('selection_mode'),
        ),
    )

    registry.register_keybinding_rule(
        '-redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyCode.KeyY,
            weight=KeyBindingWeights.PLUGIN,
            when=parse_expression('alternative_redo'),
        ),
    )

    entries = registry.get_non_canceling_entries('select_all')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(KeyCode.KeyA)

    entries = registry.get_non_canceling_entries('redo')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ
    )

    # USER ENTRIES
    registry.register_keybinding_rule(
        '-select_all',
        KeyBindingRule(
            primary=KeyCode.KeyA,
            weight=KeyBindingWeights.USER,
            when=parse_expression('selection_mode'),
        ),
    )

    registry.register_keybinding_rule(
        'select_all',
        KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.USER,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        '-redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ,
            weight=KeyBindingWeights.USER,
            when=parse_expression('~alternative_redo'),
        ),
    )

    registry.register_keybinding_rule(
        'redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ,
            weight=KeyBindingWeights.USER,
            when=None,
        ),
    )

    entries = registry.get_non_canceling_entries('select_all')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(
        KeyMod.Shift | KeyCode.KeyA
    )
    assert entries[0].weight == KeyBindingWeights.USER
    assert entries[0].when is None

    entries = registry.get_non_canceling_entries('redo')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ
    )
    assert entries[0].weight == KeyBindingWeights.USER
    assert entries[0].when is None


def test_registry_discard():
    registry = NapariKeyBindingsRegistry()

    # CORE
    registry.register_keybinding_rule(
        'select_all',
        KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.CORE,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        'undo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyCode.KeyZ,
            weight=KeyBindingWeights.CORE,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        'redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ,
            weight=KeyBindingWeights.CORE,
            when=parse_expression('~alternative_redo'),
        ),
    )

    registry.register_keybinding_rule(
        'redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyCode.KeyY,
            weight=KeyBindingWeights.CORE,
            when=parse_expression('alternative_redo'),
        ),
    )

    # PLUGIN
    registry.register_keybinding_rule(
        '-select_all',
        KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.PLUGIN,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        'select_all',
        KeyBindingRule(
            primary=KeyCode.KeyA,
            weight=KeyBindingWeights.PLUGIN,
            when=parse_expression('selection_mode'),
        ),
    )

    registry.register_keybinding_rule(
        '-redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyCode.KeyY,
            weight=KeyBindingWeights.PLUGIN,
            when=parse_expression('alternative_redo'),
        ),
    )

    # USER
    registry.register_keybinding_rule(
        '-select_all',
        KeyBindingRule(
            primary=KeyCode.KeyA,
            weight=KeyBindingWeights.USER,
            when=parse_expression('selection_mode'),
        ),
    )

    registry.register_keybinding_rule(
        'select_all',
        KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.USER,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        '-redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ,
            weight=KeyBindingWeights.USER,
            when=parse_expression('~alternative_redo'),
        ),
    )

    registry.register_keybinding_rule(
        'redo',
        KeyBindingRule(
            primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ,
            weight=KeyBindingWeights.USER,
            when=None,
        ),
    )

    # should only have core + plugin entries
    registry.discard_entries(KeyBindingWeights.USER)

    entries = registry.get_non_canceling_entries('select_all')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(KeyCode.KeyA)

    entries = registry.get_non_canceling_entries('redo')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ
    )

    # should only have core entries
    registry.discard_entries(KeyBindingWeights.PLUGIN)

    entries = registry.get_non_canceling_entries('select_all')
    assert len(entries) == 1
    assert entries[0].keybinding == KeyBinding.from_int(
        KeyMod.Shift | KeyCode.KeyA
    )

    entries = registry.get_non_canceling_entries('undo')
    assert len(entries) == 1
    assert entries[0].command_id == 'undo'

    entries = registry.get_non_canceling_entries('redo')
    assert len(entries) == 2
    assert {entry.keybinding for entry in entries} == {
        KeyBinding.from_int(KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ),
        KeyBinding.from_int(KeyMod.CtrlCmd | KeyCode.KeyY),
    }

    # should have no entries
    registry.discard_entries(KeyBindingWeights.CORE)

    assert registry.keymap == {}
