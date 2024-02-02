from unittest.mock import Mock

from app_model.expressions import Context, parse_expression
from app_model.types import KeyBindingRule, KeyChord, KeyCode, KeyMod

from napari.utils.key_bindings.constants import (
    DispatchFlags,
    KeyBindingWeights,
)
from napari.utils.key_bindings.dispatch import (
    KeyBindingDispatcher,
    find_active_match,
    has_conflicts,
    next_active_match,
)
from napari.utils.key_bindings.register import (
    KeyBindingEntry,
    NapariKeyBindingsRegistry,
)


def test_next_active_match():
    entries = [
        KeyBindingEntry(
            command_id='a',
            weight=KeyBindingWeights.CORE,
            when=None,  # always active
        ),
        KeyBindingEntry(
            command_id='b',
            weight=KeyBindingWeights.PLUGIN,
            when=parse_expression('b_active'),
        ),
        KeyBindingEntry(
            command_id='',
            weight=KeyBindingWeights.CORE,
            when=parse_expression('core_block'),
        ),
        KeyBindingEntry(
            command_id='c',
            weight=KeyBindingWeights.USER,
            when=parse_expression('c_active'),
        ),
        KeyBindingEntry(
            command_id='-b',
            weight=KeyBindingWeights.CORE,
            when=parse_expression('b_low_prio_negate'),
        ),
        KeyBindingEntry(
            command_id='-b',
            weight=KeyBindingWeights.PLUGIN,
            when=parse_expression('b_negate'),
        ),
        KeyBindingEntry(
            command_id='',
            weight=KeyBindingWeights.USER,
            when=parse_expression('user_block'),
        ),
    ]

    entries.sort()

    context = {
        'b_active': False,
        'core_block': False,
        'c_active': False,
        'b_low_prio_negate': False,
        'b_negate': False,
        'user_block': False,
    }

    # only active command
    assert [
        entry.command_id for entry in next_active_match(entries, context)
    ] == ['a']

    # activate command b
    context['b_active'] = True
    assert [
        entry.command_id for entry in next_active_match(entries, context)
    ] == ['b', 'a']

    # negation rule is lower weight than command b entry so it does nothing
    context['b_low_prio_negate'] = True
    assert [
        entry.command_id for entry in next_active_match(entries, context)
    ] == ['b', 'a']

    # negation rule removes b from the list
    context['b_negate'] = True
    assert [
        entry.command_id for entry in next_active_match(entries, context)
    ] == ['a']

    # block core commands (e.g. command a) and activate command c
    context['c_active'] = True
    context['core_block'] = True
    assert [
        entry.command_id for entry in next_active_match(entries, context)
    ] == ['c']

    # block all commands
    context['user_block'] = True
    assert [
        entry.command_id for entry in next_active_match(entries, context)
    ] == []


def test_find_active_match():
    entries = [
        KeyBindingEntry(
            command_id='a',
            weight=KeyBindingWeights.CORE,
            when=None,
        ),
        KeyBindingEntry(
            command_id='b',
            weight=KeyBindingWeights.PLUGIN,
            when=None,
        ),
        KeyBindingEntry(
            command_id='',
            weight=KeyBindingWeights.USER,
            when=parse_expression('block_all'),
        ),
    ]
    entries.sort()

    assert (
        find_active_match(entries, context={'block_all': False}).command_id
        == 'b'
    )

    assert find_active_match(entries, context={'block_all': True}) is None


def test_has_conflicts():
    # note: more robust testing for conflict filter in ../test_util:test_conflict_filter

    keymap = {
        KeyCode.KeyA: [
            KeyBindingEntry(command_id='a', weight=KeyBindingWeights.CORE)
        ],
        KeyMod.Shift
        | KeyCode.KeyA: [
            KeyBindingEntry(command_id='b', weight=KeyBindingWeights.USER),
            KeyBindingEntry(
                command_id='',
                weight=KeyBindingWeights.USER,
                when=parse_expression('block'),
            ),
        ],
    }

    assert (
        has_conflicts(KeyCode.KeyA, keymap, context={'block': False}) is False
    )

    assert (
        has_conflicts(KeyMod.Shift, keymap, context={'block': False}) is True
    )

    assert (
        has_conflicts(KeyMod.Shift, keymap, context={'block': True}) is False
    )


def test_dispatcher():
    registry = NapariKeyBindingsRegistry()
    context = Context(
        {'single_mod': False, 'block_shift_a': False, 'chord_off': False}
    )
    dispatcher = KeyBindingDispatcher(registry, context)

    registry.register_keybinding_rule(
        command_id='single_mod',
        rule=KeyBindingRule(
            primary=KeyMod.Shift,
            weight=KeyBindingWeights.USER,
            when=parse_expression('single_mod'),
        ),
    )

    registry.register_keybinding_rule(
        command_id='shift_a',
        rule=KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.CORE,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        command_id='',
        rule=KeyBindingRule(
            primary=KeyMod.Shift | KeyCode.KeyA,
            weight=KeyBindingWeights.USER,
            when=parse_expression('block_shift_a'),
        ),
    )

    registry.register_keybinding_rule(
        command_id='chord',
        rule=KeyBindingRule(
            primary=KeyChord(
                KeyMod.Shift | KeyCode.KeyA, KeyMod.Shift | KeyCode.KeyS
            ),
            weight=KeyBindingWeights.CORE,
            when=None,
        ),
    )

    registry.register_keybinding_rule(
        command_id='-chord',
        rule=KeyBindingRule(
            primary=KeyChord(
                KeyMod.Shift | KeyCode.KeyA, KeyMod.Shift | KeyCode.KeyS
            ),
            weight=KeyBindingWeights.CORE,
            when=parse_expression('chord_off'),
        ),
    )

    # TEST CACHE CLEAR
    # registration clear
    assert dispatcher.has_conflicts(KeyCode.KeyA) is False
    assert dispatcher._conflicts_cache[KeyCode.KeyA] is False

    dispose = registry.register_keybinding_rule(
        command_id='null',
        rule=KeyBindingRule(
            primary=KeyCode.KeyB, weight=KeyBindingWeights.USER, when=None
        ),
    )

    assert dispatcher._conflicts_cache == {}

    # disposal clear
    assert (
        dispatcher.find_active_match(KeyMod.Shift | KeyCode.KeyA).command_id
        == 'shift_a'
    )
    assert (
        dispatcher._active_match_cache[KeyMod.Shift | KeyCode.KeyA].command_id
        == 'shift_a'
    )

    dispose()
    assert dispatcher._active_match_cache == {}

    # context change clear
    assert dispatcher.active_keymap == {
        KeyMod.Shift | KeyCode.KeyA: 'shift_a',
        KeyChord(
            KeyMod.Shift | KeyCode.KeyA, KeyMod.Shift | KeyCode.KeyS
        ): 'chord',
    }

    context['chord_off'] = True

    assert dispatcher.active_keymap == {
        KeyMod.Shift | KeyCode.KeyA: 'shift_a',
    }

    # TEST DISPATCH
    mock = Mock()
    dispatcher.dispatch.connect(mock)

    # normal dispatch (chord_off already active)
    dispatcher.on_key_press(KeyMod.Shift, KeyCode.KeyA)
    mock.assert_called_with(DispatchFlags.RESET, 'shift_a')

    # single mod dispatch
    context['single_mod'] = True
    dispatcher.on_key_press(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_with(
        DispatchFlags.SINGLE_MOD | DispatchFlags.DELAY, 'single_mod'
    )

    dispatcher.on_key_release(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_with(
        DispatchFlags.ON_RELEASE | DispatchFlags.SINGLE_MOD, 'single_mod'
    )

    # single mod dispatch no conflicts
    context['block_shift_a'] = True
    dispatcher.on_key_press(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_with(DispatchFlags.SINGLE_MOD, 'single_mod')

    # try a key chord
    context['chord_off'] = False
    context['block_shift_a'] = False
    context['single_mod'] = False

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_once_with(DispatchFlags.RESET, None)

    # note: key combo `shift+a`` is not triggered because the chord `shift+a shift+s` takes priority
    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.Shift, KeyCode.KeyA)
    mock.assert_called_once_with(DispatchFlags.RESET, None)

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_once_with(DispatchFlags.TWO_PART, None)

    mock.reset_mock()
    dispatcher.on_key_release(KeyMod.Shift, KeyCode.KeyA)
    mock.assert_not_called()

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.Shift, KeyCode.KeyS)
    mock.assert_called_once_with(DispatchFlags.TWO_PART, 'chord')

    mock.reset_mock()
    dispatcher.on_key_release(KeyMod.NONE, KeyCode.Shift)
    mock.assert_not_called()

    mock.reset_mock()
    dispatcher.on_key_release(KeyMod.NONE, KeyCode.KeyS)
    mock.assert_called_once_with(
        DispatchFlags.ON_RELEASE | DispatchFlags.TWO_PART, 'chord'
    )

    # test again but with conflicts
    context['single_mod'] = True
    context['block_shift_a'] = False

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_once_with(
        DispatchFlags.SINGLE_MOD | DispatchFlags.DELAY, 'single_mod'
    )

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.Shift, KeyCode.KeyA)
    mock.assert_called_once_with(DispatchFlags.RESET, None)

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_once_with(DispatchFlags.TWO_PART, None)

    mock.reset_mock()
    dispatcher.on_key_release(KeyMod.Shift, KeyCode.KeyA)
    mock.assert_not_called()

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.Shift, KeyCode.KeyS)
    mock.assert_called_once_with(DispatchFlags.TWO_PART, 'chord')

    # try different order of releasing the keys
    mock.reset_mock()
    dispatcher.on_key_release(KeyMod.Shift, KeyCode.KeyS)
    mock.assert_called_with(
        DispatchFlags.ON_RELEASE | DispatchFlags.TWO_PART, 'chord'
    )

    mock.reset_mock()
    dispatcher.on_key_release(KeyMod.NONE, KeyCode.Shift)
    mock.assert_not_called()

    # disable chord to use normal key combo
    context['chord_off'] = True

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.NONE, KeyCode.Shift)
    mock.assert_called_once_with(
        DispatchFlags.SINGLE_MOD | DispatchFlags.DELAY, 'single_mod'
    )

    mock.reset_mock()
    dispatcher.on_key_press(KeyMod.Shift, KeyCode.KeyA)
    mock.assert_called_once_with(DispatchFlags.RESET, 'shift_a')

    mock.reset_mock()
    dispatcher.on_key_release(KeyMod.NONE, KeyCode.KeyA)
    mock.assert_called_with(DispatchFlags.ON_RELEASE, 'shift_a')
