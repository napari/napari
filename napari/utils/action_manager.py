from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from inspect import isgeneratorfunction
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..utils.events import EmitterGroup
from .interactions import Shortcut
from .translations import trans

if TYPE_CHECKING:
    from typing import Protocol

    from .key_bindings import KeymapProvider

    class SignalInstance(Protocol):
        def connect(self, callback: Callable) -> None:
            ...

    class Button(Protocol):
        clicked: SignalInstance

        def setToolTip(self, text: str) -> None:
            ...

    class ShortcutEvent:
        name: str
        shortcut: str
        tooltip: str


@dataclass
class Action:
    command: Callable
    description: str
    keymapprovider: KeymapProvider  # subclassclass or instance of a subclass

    @cached_property
    def injected(self) -> Callable:
        """command with napari objects injected.

        This will inject things like the current viewer, or currently selected
        layer into the commands.  See :func:`inject_napari_dependencies` for
        details.
        """
        from .._app_model import get_app

        return get_app().injection_store.inject(self.command)


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
        self._shortcuts: Dict[str, List[str]] = defaultdict(list)
        self._stack: List[str] = []
        self._tooltip_include_action_name = False
        self.events = EmitterGroup(source=self, shorcut_changed=None)

    def _debug(self, val):
        self._tooltip_include_action_name = val

    def _validate_action_name(self, name):
        if len(name.split(':')) != 2:
            raise ValueError(
                trans._(
                    'Action names need to be in the form `package:name`, got {name!r}',
                    name=name,
                    deferred=True,
                )
            )

    def register_action(
        self,
        name: str,
        command: Callable,
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

    def _update_shortcut_bindings(self, name: str):
        """
        Update the key mappable for given action name
        to trigger the action within the given context and
        """
        if name not in self._actions:
            return
        if name not in self._shortcuts:
            return
        action = self._actions[name]
        km_provider: KeymapProvider = action.keymapprovider
        if hasattr(km_provider, 'bind_key'):
            for shortcut in self._shortcuts[name]:
                # NOTE: it would be better if we could bind `self.trigger` here
                # as it allow the action manager to be a convenient choke point
                # to monitor all commands (useful for undo/redo, etc...), but
                # the generator pattern in the keybindings caller makes that
                # difficult at the moment, since `self.trigger(name)` is not a
                # generator function (but action.injected is)
                km_provider.bind_key(shortcut, action.injected, overwrite=True)

    def bind_button(
        self, name: str, button: Button, extra_tooltip_text=''
    ) -> None:
        """
        Bind `button` to trigger Action `name` on click.

        Parameters
        ----------
        name : str
            name of the corresponding action in the form ``packagename:name``
        button : Button
            A object providing Button interface (like QPushButton) that, when
            clicked, should trigger the action. The tooltip will be set to the
            action description and the corresponding shortcut if available.
        extra_tooltip_text : str
            Extra text to add at the end of the tooltip. This is useful to
            convey more information about this action as the action manager may
            update the tooltip based on the action name.

        Notes
        -----
        calling `bind_button` can be done before an action with the
        corresponding name is registered, in which case the effect will be
        delayed until the corresponding action is registered.

        Note: this method cannot be used with generator functions,
        see https://github.com/napari/napari/issues/4164 for details.
        """
        self._validate_action_name(name)

        if action := self._actions.get(name):
            if isgeneratorfunction(action):
                raise ValueError(
                    'bind_button cannot be used with generator functions'
                )

        button.clicked.connect(lambda: self.trigger(name))
        button.setToolTip(f'{self._build_tooltip(name)} {extra_tooltip_text}')

        def _update_tt(event: ShortcutEvent):
            if event.name == name:
                button.setToolTip(f'{event.tooltip} {extra_tooltip_text}')

        # if it's a QPushbutton, we'll remove it when it gets destroyed
        until = getattr(button, 'destroyed', None)
        self.events.shorcut_changed.connect(_update_tt, until=until)

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
        if shortcut in self._shortcuts[name]:
            return
        self._shortcuts[name].append(shortcut)
        self._update_shortcut_bindings(name)
        self._emit_shortcut_change(name, shortcut)

    def unbind_shortcut(self, name: str) -> Optional[List[str]]:
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
                    "Attempting to unbind an action which does not exists ({name}), this may have no effects. This can happen if your settings are out of date, if you upgraded napari, upgraded or deactivated a plugin, or made a typo in in your custom keybinding.",
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

        self._emit_shortcut_change(name)
        return shortcuts

    def _emit_shortcut_change(self, name: str, shortcut=''):
        tt = self._build_tooltip(name) if name in self._actions else ''
        self.events.shorcut_changed(name=name, shortcut=shortcut, tooltip=tt)

    def _build_tooltip(self, name: str) -> str:
        """Build tooltip for action `name`."""
        ttip = self._actions[name].description

        if name in self._shortcuts:
            jstr = ' ' + trans._p('<keysequence> or <keysequence>', 'or') + ' '
            shorts = jstr.join(f"{Shortcut(s)}" for s in self._shortcuts[name])
            ttip += f' ({shorts})'

        ttip += f'[{name}]' if self._tooltip_include_action_name else ''
        return ttip

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
            for name, shortcuts in self._shortcuts.items():
                action = self._actions.get(name, None)
                if action and layer == action.keymapprovider:
                    for shortcut in shortcuts:
                        layer_shortcuts[layer][
                            str(shortcut)
                        ] = action.description

        return layer_shortcuts

    def _get_layer_actions(self, layer) -> dict:
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
        return {
            name: action
            for name, action in self._actions.items()
            if action and layer == action.keymapprovider
        }

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
        for name, shortcuts in self._shortcuts.items():
            action = self._actions.get(name, None)
            if action and action.command.__name__ in active_func_names:
                for shortcut in shortcuts:
                    active_shortcuts[str(shortcut)] = action.description

        return active_shortcuts

    def trigger(self, name: str) -> Any:
        """Trigger the action `name`."""
        return self._actions[name].injected()


action_manager = ActionManager()
