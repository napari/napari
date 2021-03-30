"""Key combinations are represented in the form ``[modifier-]key``,
e.g. ``a``, ``Control-c``, or ``Control-Alt-Delete``.
Valid modifiers are Control, Alt, Shift, and Meta.

Letters will always be read as upper-case.
Due to the native implementation of the key system, Shift pressed in
certain key combinations may yield inconsistent or unexpected results.
Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
Control is swapped with Meta such that pressing Command reads as Control.

Special keys include Shift, Control, Alt, Meta, Up, Down, Left, Right,
PageUp, PageDown, Insert, Delete, Home, End, Escape, Backspace, F1,
F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, Space, Enter, and Tab

Functions take in only one argument: the parent that the function
was bound to.

By default, all functions are assumed to work on key presses only,
but can be denoted to work on release too by separating the function
into two statements with the yield keyword::

    @viewer.bind_key('h')
    def hello_world(viewer):
        # on key press
        viewer.status = 'hello world!'

        yield

        # on key release
        viewer.status = 'goodbye world :('

To create a keymap that will block others, ``bind_key(..., ...)```.
"""

import inspect
import re
import types
from collections import ChainMap, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict

from qtpy.QtGui import QKeySequence
from vispy.util import keys

from .interactions import format_shortcut

SPECIAL_KEYS = [
    keys.SHIFT,
    keys.CONTROL,
    keys.ALT,
    keys.META,
    keys.UP,
    keys.DOWN,
    keys.LEFT,
    keys.RIGHT,
    keys.PAGEUP,
    keys.PAGEDOWN,
    keys.INSERT,
    keys.DELETE,
    keys.HOME,
    keys.END,
    keys.ESCAPE,
    keys.BACKSPACE,
    keys.F1,
    keys.F2,
    keys.F3,
    keys.F4,
    keys.F5,
    keys.F6,
    keys.F7,
    keys.F8,
    keys.F9,
    keys.F10,
    keys.F11,
    keys.F12,
    keys.SPACE,
    keys.ENTER,
    keys.TAB,
]

MODIFIER_KEYS = [keys.CONTROL, keys.ALT, keys.SHIFT, keys.META]


def parse_key_combo(key_combo):
    """Parse a key combination into its components in a comparable format.

    Parameters
    ----------
    key_combo : str
        Key combination.

    Returns
    -------
    key : str
        Base key of the combination.
    modifiers : set of str
        Modifier keys of the combination.
    """
    parsed = re.split('-(?=.+)', key_combo)
    *modifiers, key = parsed

    return key, set(modifiers)


def components_to_key_combo(key, modifiers):
    """Combine components to become a key combination.

    Modifier keys will always be combined in the same order:
    Control, Alt, Shift, Meta

    Letters will always be read as upper-case.
    Due to the native implementation of the key system, Shift pressed in
    certain key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
    Control is swapped with Meta such that pressing Command reads as Control.

    Parameters
    ----------
    key : str or vispy.app.Key
        Base key.
    modifiers : combination of str or vispy.app.Key
        Modifier keys.

    Returns
    -------
    key_combo : str
        Generated key combination.
    """
    if len(key) == 1 and key.isalpha():  # it's a letter
        key = key.upper()
        cond = lambda m: True  # noqa: E731
    elif key in SPECIAL_KEYS:
        # remove redundant information i.e. an output of 'Shift-Shift'
        cond = lambda m: m != key  # noqa: E731
    else:
        # Shift is consumed to transform key

        # bug found on OSX: Command will cause Shift to not
        # transform the key so do not consume it
        # note: 'Control' is OSX Command key
        cond = lambda m: m != 'Shift' or 'Control' in modifiers  # noqa: E731

    modifiers = tuple(
        key.name
        for key in filter(
            lambda key: key in modifiers and cond(key), MODIFIER_KEYS
        )
    )

    return '-'.join(modifiers + (key,))


def normalize_key_combo(key_combo):
    """Normalize key combination to make it easily comparable.

    All aliases are converted and modifier orders are fixed to:
    Control, Alt, Shift, Meta

    Letters will always be read as upper-case.
    Due to the native implementation of the key system, Shift pressed in
    certain key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
    Control is swapped with Meta such that pressing Command reads as Control.

    Parameters
    ----------
    key_combo : str
        Key combination.

    Returns
    -------
    normalized_key_combo : str
        Normalized key combination.
    """
    key, modifiers = parse_key_combo(key_combo)

    if len(key) != 1 and key not in SPECIAL_KEYS:
        raise TypeError(f'invalid key {key}')

    for modifier in modifiers:
        if modifier not in MODIFIER_KEYS:
            raise TypeError(f'invalid modifier key {modifier}')

    return components_to_key_combo(key, modifiers)


UNDEFINED = object()


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
    from inspect import signature

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
    keymappable: Any


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
        self._actions = {}
        self._buttons = defaultdict(lambda: [])
        self._qactions = defaultdict(lambda: [])
        self._shortcuts = {}
        self._orphans = {}
        self.context = {}

    def register_action(self, name, command, description, keymappable):
        """
        Register an action for future usage

        An action is generally a callback associated with
         - a name (unique),
         - a description
         - A keymapable (easier for focus and backward compatibility).

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
        assert name not in self._actions, name
        self._actions[name] = Action(command, description, keymappable)
        self._update_shortcut_bindings(name)
        # shortcuts may have been bound before the command was actually
        # registered, remove it from orphan and bind shortcuts.
        if command in self._orphans:
            # print('Found orphan shortcut')
            sht = self._orphans.pop(command)
            self.bind_shortcut(name, sht)
        self._update_gui_elements(name)

    def _update_gui_elements(self, name):
        """
        Update the description and shortcuts of all the (known) gui elements.
        """
        if name not in self._actions:
            return
        buttons = self._buttons[name]
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
        if name in self._qactions:
            action = self._actions[name]
            qaction = self._qactions[name]
            qaction.setShortcut(
                QKeySequence(sht.replace('-', '+').replace('Control', 'Ctrl'))
            )
            qaction.setText(action.description)
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
        keymappable = action.keymappable
        if hasattr(keymappable, 'bind_key'):

            def flash(*args):
                call_with_context(action.command, self.context)
                for b in self._buttons.get(name, []):
                    b._animation.start()

            keymappable.bind_key(sht)(flash)

    def bind_button(self, name, button):
        """
        Bind `button` to trigger Action `name` on click.
        """
        action = self._actions[name]

        button.clicked.connect(
            lambda: call_with_context(action.command, self.context)
        )

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
                b._animation.start()

        qaction.triggered.connect(flash)
        self._update_gui_elements(name)

    def unbind_shortcut(self, name):
        """
        unbind shortcut for action name
        """
        action = self._actions[name]
        sht = self._shortcuts.get(name)
        if hasattr(action.keymappable, 'bind_key'):
            action.keymappable.bind_key(sht)(None)
        del self._shortcuts[name]
        self._update_gui_elements(name)


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
    'Ctrl+Shift+C': 'toggle_console_visibility',
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


def bind_key(keymap, key, func=UNDEFINED, *, overwrite=False):
    """Bind a key combination to a keymap.

    Parameters
    ----------
    keymap : dict of str: callable
        Keymap to modify.
    key : str or ...
        Key combination.
        ``...`` acts as a wildcard if no key combinations can be matched
        in the keymap (this will overwrite all key combinations
        further down the lookup chain).
    func : callable, None, or ...
        Callable to bind to the key combination.
        If ``None`` is passed, unbind instead.
        ``...`` acts as a blocker, effectively unbinding the key
        combination for all keymaps further down the lookup chain.
    overwrite : bool, keyword-only, optional
        Whether to overwrite the key combination if it already exists.

    Returns
    -------
    unbound : callable or None
        Callable unbound by this operation, if any.

    Notes
    -----
    Key combinations are represented in the form ``[modifier-]key``,
    e.g. ``a``, ``Control-c``, or ``Control-Alt-Delete``.
    Valid modifiers are Control, Alt, Shift, and Meta.

    Letters will always be read as upper-case.
    Due to the native implementation of the key system, Shift pressed in
    certain key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
    Control is swapped with Meta such that pressing Command reads as Control.

    Special keys include Shift, Control, Alt, Meta, Up, Down, Left, Right,
    PageUp, PageDown, Insert, Delete, Home, End, Escape, Backspace, F1,
    F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, Space, Enter, and Tab

    Functions take in only one argument: the parent that the function
    was bound to.

    By default, all functions are assumed to work on key presses only,
    but can be denoted to work on release too by separating the function
    into two statements with the yield keyword::

        @viewer.bind_key('h')
        def hello_world(viewer):
            # on key press
            viewer.status = 'hello world!'

            yield

            # on key release
            viewer.status = 'goodbye world :('

    To create a keymap that will block others, ``bind_key(..., ...)```.
    """
    if func is UNDEFINED:

        def inner(func):
            bind_key(keymap, key, func, overwrite=overwrite)
            return func

        return inner
    assert key is not Ellipsis
    if key is not Ellipsis:
        key = normalize_key_combo(key)

    if func is not None and key in keymap and not overwrite:
        raise ValueError(
            f'key combination {key} already used! '
            "specify 'overwrite=True' to bypass this check"
        )

    unbound = keymap.pop(key, None)

    if func is not None:
        if func is not Ellipsis and not callable(func):
            raise TypeError("'func' must be a callable")
        keymap[key] = func

    return unbound


class KeybindingDescriptor:
    """Descriptor which transforms ``func`` into a method with the first
    argument bound to ``class_keymap`` or ``keymap`` depending on if it was
    called from the class or the instance, respectively.

    Parameters
    ----------
    func : callable
        Function to bind.
    """

    def __init__(self, func):
        self.__func__ = func

    def __get__(self, instance, cls):
        if instance is not None:
            keymap = instance.keymap
        else:
            keymap = cls.class_keymap

        return types.MethodType(self.__func__, keymap)


class KeymapProvider:
    """Mix-in to add keymap functionality.

    Attributes
    ----------
    class_keymap : dict
        Class keymap.
    keymap : dict
        Instance keymap.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keymap = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if 'class_keymap' not in cls.__dict__:
            # if in __dict__, was defined in class and not inherited
            cls.class_keymap = {}

    bind_key = KeybindingDescriptor(bind_key)

    @classmethod
    def register_action(cls, *, description=None, name=None):
        """
        Convenient decorator to register an action with the current KeymapProvider

        It will use the function name as the action name, and the function docstring as
        the function description.
        """

        def _inner(func):
            nonlocal name
            nonlocal description
            if name is None:
                name = func.__name__
            assert name is not None
            if description is None:
                description = func.__doc__
            action_manager.register_action(name, func, description, cls)
            return func

        return _inner


def _bind_keymap(keymap, instance):
    """Bind all functions in a keymap to an instance.

    Parameters
    ----------
    keymap : dict
        Keymap to bind.
    instance : object
        Instance to bind to.

    Returns
    -------
    bound_keymap : dict
        Keymap with functions bound to the instance.
    """
    bound_keymap = {
        key: types.MethodType(func, instance) if func is not Ellipsis else func
        for key, func in keymap.items()
    }
    return bound_keymap


class KeymapHandler:
    """Handle key mapping and calling functionality.

    Attributes
    ----------
    keymap_providers : list of KeymapProvider
        Classes that provide the keymaps for this class to handle.
    """

    def __init__(self):
        super().__init__()
        self._key_release_generators = {}
        self.keymap_providers = []

    @property
    def keymap_chain(self):
        """collections.ChainMap: Chain of keymaps from keymap providers."""
        maps = []

        for parent in self.keymap_providers:
            maps.append(_bind_keymap(parent.keymap, parent))
            # For parent and superclasses add inherited keybindings
            for cls in parent.__class__.__mro__:
                if hasattr(cls, 'class_keymap'):
                    maps.append(_bind_keymap(cls.class_keymap, parent))

        return ChainMap(*maps)

    @property
    def active_keymap(self):
        """dict: Active keymap, created by resolving the keymap chain."""
        active_keymap = self.keymap_chain
        keymaps = active_keymap.maps

        for i, keymap in enumerate(keymaps):
            if Ellipsis in keymap:  # catch-all key
                # trim all keymaps after catch-all
                active_keymap = ChainMap(*keymaps[: i + 1])
                break

        active_keymap_final = {
            k: func
            for k, func in active_keymap.items()
            if func is not Ellipsis
        }

        return active_keymap_final

    def press_key(self, key_combo):
        """Simulate a key press to activate a keybinding.

        Parameters
        ----------
        key_combo : str
            Key combination.
        """
        key_combo = normalize_key_combo(key_combo)
        keymap = self.active_keymap
        if key_combo in keymap:
            func = keymap[key_combo]
        elif Ellipsis in keymap:  # catch-all
            func = keymap[...]
        else:
            return  # no keybinding found

        if func is Ellipsis:  # blocker
            return
        elif not callable(func):
            raise TypeError(f"expected {func} to be callable")

        gen = func()

        if inspect.isgeneratorfunction(func):
            try:
                next(gen)  # call function
            except StopIteration:  # only one statement
                pass
            else:
                key, _ = parse_key_combo(key_combo)
                self._key_release_generators[key] = gen

    def release_key(self, key_combo):
        """Simulate a key release for a keybinding.

        Parameters
        ----------
        key_combo : str
            Key combination.
        """
        key, _ = parse_key_combo(key_combo)
        try:
            next(self._key_release_generators[key])  # call function
        except (KeyError, StopIteration):
            pass

    def on_key_press(self, event):
        """Callback that whenever key pressed in canvas.

        Parameters
        ----------
        event : vispy.util.event.Event
            The vispy key press event that triggered this method.
        """
        if (
            event.native is not None
            and event.native.isAutoRepeat()
            and event.key.name not in ['Up', 'Down', 'Left', 'Right']
        ) or event.key is None:
            # pass if no key is present or if key is held down, unless the
            # key being held down is one of the navigation keys
            # this helps for scrolling, etc.
            return

        combo = components_to_key_combo(event.key.name, event.modifiers)
        self.press_key(combo)

    def on_key_release(self, event):
        """Called whenever key released in canvas.

        Parameters
        ----------
        event : vispy.util.event.Event
            The vispy key release event that triggered this method.
        """
        if event.key is None or (
            # on linux press down is treated as multiple press and release
            event.native is not None
            and event.native.isAutoRepeat()
        ):
            return
        combo = components_to_key_combo(event.key.name, event.modifiers)
        self.release_key(combo)
