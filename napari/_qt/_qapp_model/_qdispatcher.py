from __future__ import annotations

import time
from functools import partial
from typing import Callable, Optional

from app_model.registries import CommandsRegistry
from qtpy.QtCore import QObject, QTimer

from napari._app_model.actions import AttrRestoreCallback, GeneratorCallback
from napari.settings import get_settings
from napari.utils.key_bindings import DispatchFlags

# TODO: make this a user setting
PRESS_HOLD_DELAY_MS = 200


class QKeyBindingDispatcher(QObject):
    def __init__(
        self,
        commands: CommandsRegistry,
        check_repeatable_action: Callable[[str], bool],
        *,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._commands = commands
        self._action_is_repeatable = check_repeatable_action
        self.timer: Optional[QTimer] = None
        self._attr_restore_start_time: float = 0
        self._attr_restore_callback = None

    def executeCommand(
        self, command_id: str, *, on_press: bool = False
    ) -> None:
        func = self._commands[command_id].resolved_callback

        if isinstance(func, AttrRestoreCallback):
            if on_press:
                self._attr_restore_callback = self._commands.execute_command(
                    command_id
                ).result()
                self._attr_restore_start_time = time.time()
            elif (
                self._attr_restore_callback is not None
                and time.time() - self._attr_restore_start_time
                > get_settings().application.hold_button_delay
            ):
                self._attr_restore_callback()
                self._attr_restore_callback = None
                self._attr_restore_start_time = 0
            return

        if isinstance(func, GeneratorCallback):
            # has release logic too
            if on_press:
                func.reset()
            elif func._gen is None:
                return
        elif not on_press:
            return

        self._commands.execute_command(command_id)

    def onDispatch(
        self, flags: DispatchFlags, command_id: Optional[str]
    ) -> None:
        active_timer = False
        cancel_timer = False
        on_press = False

        if (
            command_id is not None
            and flags & DispatchFlags.IS_AUTO_REPEAT
            and not self._action_is_repeatable(command_id)
        ):
            # ignore repeats for non-repeatable keys
            return

        if flags == DispatchFlags.RESET:
            cancel_timer = True

        if flags & DispatchFlags.ON_RELEASE:
            if self.timer is not None:
                active_timer = self.timer.isActive()
                cancel_timer = True
        else:
            if (
                command_id
                and flags & DispatchFlags.SINGLE_MOD
                and flags & DispatchFlags.DELAY
            ):
                self.timer = QTimer(self)
                self.timer.setSingleShot(True)

                self.timer.timeout.connect(
                    partial(self.executeCommand, command_id, on_press=True)
                )

                self.timer.start(PRESS_HOLD_DELAY_MS)
                return
            cancel_timer = True
            on_press = True

        if cancel_timer and self.timer is not None:
            self.timer.stop()
            self.timer = None

        if command_id:
            if active_timer:
                # execute press and release logic at the same time
                self.executeCommand(command_id, on_press=True)
            self.executeCommand(command_id, on_press=on_press)
