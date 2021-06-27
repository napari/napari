from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, List, Set, Tuple, Union

from .interactions import Shortcut
from .key_bindings import KeymapProvider
from .translations import trans

if TYPE_CHECKING:
    from qtpy.QtWidgets import QPushButton

    from napari.qt import QtStateButton


def call_with_context(function, context):
    """
    call function `function` with the corresponding value taken from context

    This is use in the action manager to pass the rights instances to the actions,
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
    keymapprovider: KeymapProvider  # subclassclass or instance of a subclass

    def callable(self, context):
        if not hasattr(self, '_command_with_context'):
            self._command_with_context = lambda: call_with_context(
                self.command, context
            )
        return self._command_with_context


class ButtonWrapper:
    """
    Pyside seem to have an issue where calling disconnect/connect
    seem to segfault.

    We wrap our buttons in this to no call disconnect and reconnect the same callback
    When not necessary.

    This also make it simpler to make connecting a callback idempotent, and make
    it easier to re-update all gui elements when a shortcut or description is
    changed.
    """

    def __init__(self, button, extra_tooltip_text):
        """
        wrapper around button to disconnect an action only
        if it has been connected before.
        """
        self._button = button
        self._connected = None
        self._extra_tooltip_text: str = extra_tooltip_text

    def setToolTip(self, value):
        return self._button.setToolTip(value + ' ' + self._extra_tooltip_text)

    def clicked_maybe_connect(self, callback):
        if callback is not self._connected:
            if self._connected is not None:
                self._button.clicked.disconnect(self._connected)
            self._button.clicked.connect(callback)
            self._connected = callback
        else:
            # do nothing it's the same callback.
            pass

    @property
    def destroyed(self):
        return self._button.destroyed


class Context:
    def __init__(self):
        self._attr = {}
        self._callables = {}

    def __getitem__(self, key):
        if key in self._attr:
            return self._attr[key]
        elif key in self._callables:
            return self._callables[key]()

    def items(self):
        for k, v in self._attr.items():
            yield k, v
        for k, v in self._callables.items():
            yield k, v()

    def __setitem__(self, key, value):
        if callable(value):
            self._callables[key] = value
        else:
            self._attr[key] = value


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
            str, Set[Tuple(Union[QPushButton, QtStateButton], str)]
        ] = defaultdict(lambda: set())
        self._shortcuts: Dict[str, set[str]] = defaultdict(lambda: set())
        self.context = Context()  # Dict[str, Any] = {}
        self._stack: List[str] = []
        self._tooltip_include_action_name = False

    def _debug(self, val):
        self._tooltip_include_action_name = val
        for name in self._buttons.keys():
            self._update_gui_elements(name)

    def _validate_action_name(self, name):
        if ":" not in name:
            raise ValueError(
                trans._(
                    'Action names need to be in the form `package:name`, got {name}',
                    name=name,
                    deferred=True,
                )
            )

    def register_action(
        self,
        name: str,
        command: callable,
        description: str,
        keymapprovider: KeymapProvider,
    ):
        """
        Register an action for future usage

        An action is generally a callback associated with
         - a name (unique), usually `packagename:name`
         - a description
         - A keymap provider (easier for focus and backward compatibility).

        Actions can then be later bound/unbound from button elements, and
        shortcuts; and the action manager will take care of modifying the keymap
        of instances to handle shortcuts; and UI elements to have tooltips with
        descriptions and shortcuts;

        Parameters
        ----------
        name : str
            unique name/id of the command that can be used to refer to this
            command
        command : callable
            take 0, or 1 parameter; if `keymapprovider` is not None, will be
            called with `keymapprovider` as first parameter.
        description : str
            Long string to describe what the command does, will be used in
            tooltips.
        keymapprovider : KeymapProvider
            KeymapProvider class or instance to use to bind the shortcut(s) when
            registered. This make sure the shortcut is active only when an
            instance of this is in focus.

        Notes
        -----
        Registering an action, binding buttons and shortcuts can happen in any
        order and should have the same effect. In particular registering an
        action can happen later (plugin loading), while user preference
        (keyboard shortcut), has already been happen. When this is the case, the
        button and shortcut binding is delayed until an action with the
        corresponding name is registered.

        See Also
        --------
        bind_button, bind_shortcut

        """
        self._validate_action_name(name)
        self._actions[name] = Action(command, description, keymapprovider)
        self._update_shortcut_bindings(name)
        self._update_gui_elements(name)

    def _update_buttons(self, buttons, tooltip: str, callback):
        for button in buttons:
            # test if only tooltip makes crash
            button.setToolTip(tooltip)
            button.clicked_maybe_connect(callback)

    def _update_gui_elements(self, name: str):
        """
        Update the description and shortcuts of all the (known) gui elements.
        """
        if name not in self._actions:
            return
        buttons = self._buttons.get(name, set())
        desc = self._actions[name].description

        # update buttons with shortcut and description
        if name in self._shortcuts:
            shortcuts = self._shortcuts[name]
            joinstr = (
                ' '
                + trans._('or', msgctxt='<keysequence> or <keysequence>')
                + ' '
            )
            shortcut_str = (
                '('
                + joinstr.join(
                    f"{Shortcut(shortcut).platform}" for shortcut in shortcuts
                )
                + ')'
            )
        else:
            shortcut_str = ''

        callable_ = self._actions[name].callable(self.context)
        if self._tooltip_include_action_name:
            append = '[' + name + ']'
        else:
            append = ''
        self._update_buttons(buttons, desc + shortcut_str + append, callable_)

    def _update_shortcut_bindings(self, name: str):
        """
        Update the key mappable for given action name
        to trigger the action within the given context and
        """
        if name not in self._actions:
            return
        action = self._actions[name]
        if name not in self._shortcuts:
            return
        shortcuts = self._shortcuts.get(name)
        keymapprovider = action.keymapprovider
        if hasattr(keymapprovider, 'bind_key'):
            for shortcut in shortcuts:
                keymapprovider.bind_key(shortcut, overwrite=True)(
                    action.command
                )

    def bind_button(self, name: str, button, extra_tooltip_text='') -> None:
        """
        Bind `button` to trigger Action `name` on click.

        Parameters
        ----------
        name : str
            name of the corresponding action in the form ``packagename:name``
        button : QtStateButton | QPushButton
            A abject presenting a qt-button like interface that when clicked
            should trigger the action. The tooltip will be set the action
            description and the corresponding shortcut if available.
        extra_tooltip_text : str
            Extra text to add at the end of the tooltip. This is useful to
            convey more information about this action as the action manager may
            update the tooltip based on the action name.

        Notes
        -----
        calling `bind_button` can be done before an action with the
        corresponding name is registered, in which case the effect will be
        delayed until the corresponding action is registered.

        """
        self._validate_action_name(name)
        if hasattr(button, 'change'):
            button.clicked.disconnect(button.change)
        button = ButtonWrapper(button, extra_tooltip_text)
        assert button not in [x._button for x in self._buttons['name']]

        button.destroyed.connect(lambda: self._buttons[name].remove(button))
        self._buttons[name].add(button)
        self._update_gui_elements(name)

    def bind_shortcut(self, name: str, shortcut: str) -> None:
        """
        bind shortcut `shortcut` to trigger action `name`

        Parameters
        ----------
        name : str
            name of the corresponding action in the form ``packagename:name``
        shortcut : str
            Shortcut to assign to this action use dash as separator. See
            `Shortcut` for known modifiers.

        Notes
        -----
        calling `bind_button` can be done before an action with the
        corresponding name is registered, in which case the effect will be
        delayed until the corresponding action is registered.
        """
        self._validate_action_name(name)
        self._shortcuts[name].add(shortcut)
        self._update_shortcut_bindings(name)
        self._update_gui_elements(name)

    def unbind_shortcut(self, name: str) -> Union[Set[str], None]:
        """
        Unbind all shortcuts for a given action name.

        Parameters
        ----------
        name : str
            name of the action in the form `packagename:name` to unbind.

        Returns
        -------
        shortcuts: set of str | None
            Previously bound shortcuts or None if not such shortcuts was bound,
            or no such action exists.

        Warns
        -----
        UserWarning:
            When trying to unbind an action unknown form the action manager,
            this warning will be emitted.

        """
        action = self._actions.get(name, None)
        if action is None:
            warnings.warn(
                trans._(
                    "Attempting to unbind an action which does not exists ({name}), "
                    "this may have no effects. This can happen if your settings are out of "
                    "date, if you upgraded napari, upgraded or deactivated a plugin, or made "
                    "a typo in in your custom keybinding.",
                    name=name,
                ),
                UserWarning,
                stacklevel=2,
            )

        shortcuts = self._shortcuts.get(name)
        if shortcuts:
            if action and hasattr(action.keymapprovider, 'bind_key'):
                for shortcut in shortcuts:
                    action.keymapprovider.bind_key(shortcut)(None)
            del self._shortcuts[name]
        self._update_gui_elements(name)
        return shortcuts

    def _get_layer_shortcuts(self, layers):
        """
        Get shortcuts filtered by the given layers.

        Parameters
        ----------
        layers : list of layers
            Layers to use for shortcuts filtering.

        Returns
        -------
        dict
            Dictionary of layers with dictionaries of shortcuts to
            descriptions.
        """
        layer_shortcuts = {}
        for layer in layers:
            layer_shortcuts[layer] = {}
            for name, shortcut in self._shortcuts.items():
                action = self._actions.get(name, None)
                if action and layer == action.keymapprovider:
                    layer_shortcuts[layer][str(shortcut)] = action.description

        return layer_shortcuts

    def _get_layer_actions(self, layer):
        """
        Get actions filtered by the given layers.

        Parameters
        ----------
        layer : Layer
            Layer to use for actions filtering.

        Returns
        -------
        layer_actions: dict
            Dictionary of names of actions with action values for a layer.

        """
        layer_actions = {}
        for name, action in self._actions.items():
            if action and layer == action.keymapprovider:
                layer_actions[name] = action

        return layer_actions

    def _get_active_shortcuts(self, active_keymap):
        """
        Get active shortcuts for the given active keymap.

        Parameters
        ----------
        active_keymap : KeymapProvider
            The active keymap provider.

        Returns
        -------
        dict
            Dictionary of shortcuts to descriptions.
        """
        active_func_names = [i[1].__name__ for i in active_keymap.items()]
        active_shortcuts = {}
        for name, shortcut in self._shortcuts.items():
            action = self._actions.get(name, None)
            if action and action.command.__name__ in active_func_names:
                active_shortcuts[str(shortcut)] = action.description

        return active_shortcuts


action_manager = ActionManager()
