import time
from unittest.mock import patch

from app_model.registries import CommandsRegistry

from napari._qt._qapp_model._qdispatcher import (
    PRESS_HOLD_DELAY_MS,
    QKeyBindingDispatcher,
)
from napari.utils.key_bindings import DispatchFlags


@patch.object(QKeyBindingDispatcher, 'executeCommand')
def test_q_key_binding_dispatcher_simple(mock, qtbot):
    dispatcher = QKeyBindingDispatcher(
        CommandsRegistry(),
        check_repeatable_action=lambda x: x == 'repeatable',
    )

    # test auto repeat
    dispatcher.onDispatch(DispatchFlags.IS_AUTO_REPEAT, 'command')
    mock.assert_not_called()

    dispatcher.onDispatch(DispatchFlags.IS_AUTO_REPEAT, 'repeatable')
    mock.assert_called_once_with('repeatable', on_press=True)

    # SINGLE MODIFIER
    # no delay
    mock.reset_mock()
    dispatcher.onDispatch(DispatchFlags.SINGLE_MOD, 'single_mod')
    mock.assert_called_once_with('single_mod', on_press=True)

    # delay
    mock.reset_mock()
    dispatcher.onDispatch(
        DispatchFlags.SINGLE_MOD | DispatchFlags.DELAY, 'single_mod'
    )
    mock.assert_not_called()
    assert dispatcher.timer.isActive()

    with qtbot.waitSignal(
        dispatcher.timer.timeout, timeout=2 * PRESS_HOLD_DELAY_MS
    ):
        dispatcher.timer.start(PRESS_HOLD_DELAY_MS)

    assert not dispatcher.timer.isActive()

    mock.assert_called_once_with('single_mod', on_press=True)

    mock.reset_mock()
    dispatcher.onDispatch(
        DispatchFlags.SINGLE_MOD | DispatchFlags.ON_RELEASE, 'single_mod'
    )
    mock.assert_called_once_with('single_mod', on_press=False)

    # delay with immediate release
    mock.reset_mock()
    dispatcher.onDispatch(
        DispatchFlags.SINGLE_MOD | DispatchFlags.DELAY, 'single_mod'
    )
    mock.assert_not_called()
    assert dispatcher.timer.isActive()

    dispatcher.timer.start(1000)
    time.sleep(0.1)
    assert dispatcher.timer.isActive()
    dispatcher.onDispatch(
        DispatchFlags.SINGLE_MOD | DispatchFlags.ON_RELEASE, 'single_mod'
    )
    assert mock.call_args_list == [
        (('single_mod',), {'on_press': True}),
        (('single_mod',), {'on_press': False}),
    ]


# see napari/utils/key_bindings/_tests/test_dispatch:test_dispatcher
# for source of dispatch emission
@patch.object(QKeyBindingDispatcher, 'executeCommand')
def test_q_key_binding_dispatcher_scenario_1(mock, qtbot):
    dispatcher = QKeyBindingDispatcher(
        CommandsRegistry(),
        check_repeatable_action=lambda x: x == 'repeatable',
    )

    # on_key_press(KeyMod.NONE, KeyCode.Shift)
    dispatcher.onDispatch(DispatchFlags.RESET, None)
    mock.assert_not_called()

    # on_key_press(KeyMod.Shift, KeyCode.KeyA)
    dispatcher.onDispatch(DispatchFlags.RESET, None)
    mock.assert_not_called()

    # on_key_press(KeyMod.NONE, KeyCode.Shift)
    dispatcher.onDispatch(DispatchFlags.TWO_PART, None)
    mock.assert_not_called()

    # on_key_release(KeyMod.Shift, KeyCode.KeyA)
    # no dispatch emitted

    # on_key_press(KeyMod.Shift, KeyCode.KeyS)
    dispatcher.onDispatch(DispatchFlags.TWO_PART, 'chord')
    mock.assert_called_once_with('chord', on_press=True)

    # on_key_release(KeyMod.NONE, KeyCode.Shift)
    # no dispatch emitted

    mock.reset_mock()
    # on_key_release(KeyMod.NONE, KeyCode.KeyS)
    dispatcher.onDispatch(
        DispatchFlags.ON_RELEASE | DispatchFlags.TWO_PART, 'chord'
    )
    mock.assert_called_once_with('chord', on_press=False)


@patch.object(QKeyBindingDispatcher, 'executeCommand')
def test_q_key_binding_dispatcher_scenario_2(mock, qtbot):
    dispatcher = QKeyBindingDispatcher(
        CommandsRegistry(),
        check_repeatable_action=lambda x: x == 'repeatable',
    )

    # on_key_press(KeyMod.NONE, KeyCode.Shift)
    dispatcher.onDispatch(
        DispatchFlags.SINGLE_MOD | DispatchFlags.DELAY, 'single_mod'
    )
    assert dispatcher.timer.isActive()
    mock.assert_not_called()

    # on_key_press(KeyMod.Shift, KeyCode.KeyA)
    dispatcher.onDispatch(DispatchFlags.RESET, None)
    mock.assert_not_called()
    assert dispatcher.timer is None

    # on_key_release(KeyMod.Shift, KeyCode.KeyA)
    # no dispatch emitted

    # on_key_press(KeyMod.Shift, KeyCode.KeyS)
    dispatcher.onDispatch(DispatchFlags.TWO_PART, 'chord')
    mock.assert_called_once_with('chord', on_press=True)

    mock.reset_mock()
    # on_key_release(KeyMod.Shift, KeyCode.KeyS)
    dispatcher.onDispatch(
        DispatchFlags.ON_RELEASE | DispatchFlags.TWO_PART, 'chord'
    )
    mock.assert_called_once_with('chord', on_press=False)

    # on_key_release(KeyMod.NONE, KeyCode.Shift)
    # no dispatch emitted
