from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QObject, QTimer

from napari.utils.kb import DispatchFlags

# TODO: make this a user setting
PRESS_HOLD_DELAY_MS = 200


class QKeyBindingDispatcher(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

        from napari._app_model._app import get_app

        app = get_app()

        self._app = app
        self._commands = app.commands
        self.timer = None

    def executeCommand(self, command_id, on_press=False):
        func = self._commands[command_id].callback

        if getattr(func, 'GENERATOR', False):
            # has release logic too
            if on_press:
                print(f'reset {command_id}')
                func.reset()
            elif func._gen is None:
                return
        elif not on_press:
            return

        self._commands.execute_command(command_id)

    def onDispatch(self, flags: DispatchFlags, command_id: Optional[str]):
        cancel_timer = False
        on_press = False

        if (
            flags & DispatchFlags.IS_AUTO_REPEAT
            and not self._app.action_is_repeatable(command_id)
        ):
            # ignore repeats for non-repeatable keys
            return

        if flags == DispatchFlags.RESET:
            cancel_timer = True

        if flags & DispatchFlags.ON_RELEASE:
            active_timer = False
            if self.timer:
                active_timer = self.timer.isActive()
                self.timer.stop()
                self.timer = None
            if command_id and active_timer:
                # execute press and release logic at the same time
                self.executeCommand(command_id, on_press=True)
        else:
            if command_id and flags & (
                DispatchFlags.SINGLE_MOD | DispatchFlags.DELAY
            ):
                self.timer = QTimer(self)
                self.timer.setSingleShot(True)
                self.timer.timeout.connect(
                    lambda: self.executeCommand(command_id, on_press=True)
                )
                self.timer.start(PRESS_HOLD_DELAY_MS)
                return
            cancel_timer = True
            on_press = True

        if cancel_timer and self.timer is not None:
            self.timer.stop()
            self.timer = None

        if command_id:
            self.executeCommand(command_id, on_press)
