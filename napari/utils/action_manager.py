from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from .interactions import format_shortcut
from .key_bindings import KeymapProvider

if TYPE_CHECKING:
    from qtpy.QtWidgets import QAction, QPushButton

    from napari.qt import QtStateButton


def call_with_context(function, context):
    """
    call function `function` with the corresponding value taken from context

    This is use in the action manager to pass the right instances to the actions,
    without having the need for them to take a **kwarg, and is mostly needed when
    triggering actions via buttons, or to record.

    If we went a declarative way to trigger action we cannot refer to instances
    or objects that must be passed to the action, or at least this is
    problematic.

    We circumvent this by having a context (dictionary of str:instance) in
    the action manager, and anything can tell the action manager "this is the
    current instance a key". When an action is triggered; we inspect the
    signature look at which instances it may need and pass this as parameters.
    """

    context_keys = [
        k
        for k, v in signature(function).parameters.items()
        if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)
    ]
    ctx = {k: v for k, v in context.items() if k in context_keys}
    return function(**ctx)


@dataclass
class Action:
    command: Callable
    description: str
    keymapprovider: KeymapProvider

    def call(self, context):
        if not hasattr(self, '_command_with_context'):
            self._command_with_context = lambda: call_with_context(
                self.command, context
            )
        return self._command_with_context


class ActionManager:
    """
    Manage the bindings between buttons; shortcuts, callbacks gui elements...

    The action manager is aware of the various buttons, keybindings and other
    elements that may trigger an action and is able to synchronise all of those.
    Thus when a shortcut is bound; this should be capable of updating the
    buttons tooltip menus etc to show the shortcuts, descriptions...

    In most cases this should also allow to bind non existing shortcuts,
    actions, buttons, in which case they will be bound only once the actions are
    registered.

    As the action manager also knows about all the UI elements that can trigger
    an action, it can also be used to flash the buttons when the action is
    triggered separately.

    For actions that need access to a global element (a viewer, a plugin, ... ),
    you want to give this item a unique name, and add it to the action manager
    `context` object.

    >>> action_manager.context['number'] = 1
    ... action_manager.context['qtv'] = viewer.qt_viewer

    >>> def callback(qtv, number):
    ...     qtv.dims[number] +=1

    >>> action_manager.register_action('bump one', callback,
    ...     'Add one to dims',
    ...     None)

    The callback signature is going to be inspected and required globals passed
    in.
    """

    _actions: Dict[str, Action]

    def __init__(self):
        # map associating a name/id with a Comm
        self._actions: Dict[str, Action] = {}
        self._buttons: Dict[
            str, List[Union[QPushButton, QtStateButton]]
        ] = defaultdict(lambda: [])
        self._qactions: Dict[str, QAction] = defaultdict(lambda: [])
        self._shortcuts: Dict[str, str] = {}
        self.context: Dict[str, Any] = {}
        self._stack: List[str] = []
        self._recording: bool = False

    def register_action(
        self, name, command, description, keymapprovider: KeymapProvider
    ):
        """
        Register an action for future usage

        An action is generally a callback associated with
         - a name (unique),
         - a description
         - A keymap provider (easier for focus and backward compatibility).

        Actions can then be later bound/unbound from button elements, and
        shortcuts; and the action manager will take care of modifying the keymap
        of instances to handle shortcuts; and UI elements to have tooltips with
        descriptions and shortcuts;

        Parameters
        ----------

        name: str
            unique name/id of the command that can be used to refer to this
            command
        command: callable
            take 0, or 1 parameter; is `instance` is not None, will be called
            `instance` as first parameter.
        description: str
            Long string to describe what the command does, will be used in
            tooltips.

        """
        # if name in self._actions:
        #    import warnings

        #    warnings.warn(
        #        f'Warning, action {name} already exists, make sure '
        #        'you are not overwriting an existing action',
        #        stacklevel=2,
        #    )
        self._actions[name] = Action(command, description, keymapprovider)
        self._update_shortcut_bindings(name)
        self._update_gui_elements(name)

    def _update_gui_elements(self, name):
        """
        Update the description and shortcuts of all the (known) gui elements.
        """
        if name not in self._actions:
            return
        buttons = self._buttons.get(name, [])
        desc = self._actions[name].description

        # update buttons with shortcut and description
        if name in self._shortcuts:
            sht = self._shortcuts[name]
            sht_str = f' ({format_shortcut(sht)})'
        else:
            sht = ''
            sht_str = ''

        for button in buttons:
            button.setToolTip(desc + sht_str)
            action = self._actions[name]

            try:
                # not sure how to check whether things are connected already
                button.clicked.disconnect(action.call(self.context))
            except Exception:
                pass
            button.clicked.connect(action.call(self.context))
        if name in self._qactions:
            action = self._actions[name]
            qaction = self._qactions[name]
            # We can't set the shortcut yet, or this will later break the layers
            # when we use the action manager for those. Setting the key sequence
            # there will _also_ bind the shortcuts but at the window level which
            # has a higher priority. SO this conflict with KeymapProvider
            #
            # qaction.setShortcut(
            #     QKeySequence(sht.replace('-', '+').replace('Control', 'Ctrl'))
            # )
            menu_name = name
            if ' ' not in menu_name:
                components = [
                    word.capitalize() for word in menu_name.split('_')
                ]
                menu_name = ' '.join(components)
            qaction.setText(menu_name)
            qaction.setStatusTip(action.description)

    def _update_shortcut_bindings(self, name):
        """
        Update the key mappable for given action name
        to trigger the action within the given context and
        make all the corresponding UI element flash if possible.
        """
        if name not in self._actions:
            return
        action = self._actions[name]
        if name not in self._shortcuts:
            return
        sht = self._shortcuts.get(name)
        keymapprovider = action.keymapprovider
        if hasattr(keymapprovider, 'bind_key'):

            def flash(*args):

                # here we do not need call with context as we are still using
                # KeymapProvider API that will pass their own instance as the
                # first argument. This will likely need to be rethought in the
                # future.
                self.push(name)
                action.command(*args)
                for b in self._buttons.get(name, []):
                    b._flash_animation.start()

            keymapprovider.bind_key(sht)(flash)

    def push(self, name):
        if self._recording:
            self._stack.append(name)

    def bind_button(self, name, button):
        """
        Bind `button` to trigger Action `name` on click.
        """
        if hasattr(button, 'change'):
            button.clicked.disconnect(button.change)
        button.clicked.connect(lambda: self.push(name))

        button.destroyed.connect(lambda: self._buttons[name].remove(button))

        self._buttons[name].append(button)
        self._update_gui_elements(name)

    def bind_shortcut(self, name, shortcut):
        """
        bind shortcut `shortcut` to trigger action `name`
        """
        self._shortcuts[name] = shortcut
        self._update_shortcut_bindings(name)
        self._update_gui_elements(name)

    def bind_qaction(self, name, qaction):
        """
        Bind the given qaction to an action.

        This will also update the description and shortcut when those changes.

        Same as for shortcut; make ui element flash when action is triggered.
        """
        self._qactions[name] = qaction
        action = self._actions[name]

        def flash(*args):
            call_with_context(action.command, self.context)

            for b in self._buttons.get(name, []):
                b._flash_animation.start()

        qaction.destroyed.connect(lambda: self._qactions.pop(name, None))
        qaction.triggered.connect(flash)
        qaction.triggered.connect(lambda: self.push(name))
        self._update_gui_elements(name)

    def unbind_shortcut(self, name):
        """
        unbind shortcut for action name
        """
        action = self._actions[name]
        sht = self._shortcuts.get(name)
        if hasattr(action.keymapprovider, 'bind_key'):
            action.keymapprovider.bind_key(sht)(None)
        del self._shortcuts[name]
        self._update_gui_elements(name)

    def play(self, seq):
        for i, action_name in enumerate(seq):
            act = self._actions[action_name]

            for b in self._buttons.get(action_name, []):

                def loc(cmd, b):
                    def inner():
                        b._flash_animation.start()
                        call_with_context(cmd, self.context)

                    return inner

                print('scheduling in', i * 950, act.command, i)
                from qtpy import QtCore

                QtCore.QTimer.singleShot(i * 950, loc(act.command, b))


action_manager = ActionManager()


_actions = {
    'V': 'toggle_selected_visibility',
    'Control-G': 'toggle_grid',
    'Shift-Down': 'also_select_layer_below',
    'Shift-Up': 'also_select_layer_above',
    'Down': 'select_layer_below',
    'Up': 'select_layer_above',
    'Control-A': 'select_all',
    'Control-T': 'transpose_axes',
    'Control-R': 'reset_view',
    'Control-Shift-C': 'toggle_console_visibility',
    'Control-E': 'roll_axes',
    'Alt-Down': 'focus_axes_down',
    'Alt-Up': 'focus_axes_up',
    'Right': 'increment_dims_right',
    'Left': 'increment_dims_left',
    'Control-Y': 'toggle_ndisplay',
    'Control-Shift-Backspace': 'remove_all_layers',
    'Control-Shift-Delete': 'remove_all_layers',
    'Control-Backspace': 'remove_selected',
    'Control-Delete': 'remove_selected',
}

for (shortcut, action_name) in _actions.items():
    action_manager.bind_shortcut(action_name, shortcut)
