import re
import types
from collections import OrderedDict
from enum import Enum
from typing import Sequence, Iterable, Mapping, ByteString, Callable

from vispy.util import keys


STRINGTYPES = str, ByteString


SPECIAL_KEYS = [keys.SHIFT,
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
                keys.TAB]

MODIFIER_KEYS = OrderedDict(C=keys.CONTROL,
                            M=keys.ALT,
                            S=keys.SHIFT,
                            s=keys.META)


shorthand_modifier_patt = re.compile(f"([{''.join(MODIFIER_KEYS)}])(?=-)")


class BindOperation(Enum):
    BIND = 'bind'
    UNBIND = 'unbind'
    REBIND = 'rebind'


def _expand_shorthand(shorthand_seq):
    """Expand shorthand form of a key sequence.

    C -> Control
    M -> Alt
    S -> Shift
    s -> Meta
    """
    return re.sub(shorthand_modifier_patt,
                  lambda m: MODIFIER_KEYS[m.group(0)].name,
                  shorthand_seq)


def parse_seq(seq):
    """Parse a key sequence into its components in a comparable format.

    Parameters
    ----------
    seq : str
        Key sequence.

    Returns
    -------
    key : str
        Base key of the sequence.
    modifiers : set of str
        Modifier keys of the sequence.
    """
    parsed = re.split('-(?=.+)', seq)
    modifiers, key = parsed[:-1], parsed[-1]

    return key, set(modifiers)


def components_to_seq(key, modifiers):
    """Combine components to become a key sequence.

    Modifier keys will always be combined in the same order:
    Control, Alt, Shift, Meta

    Shift pressed with any letter will be consumed
    to transform it to upper case, e.g. Shift + w => W.
    Otherwise, the letter will be lower case.
    Shift will also be consumed without transformation when pressed
    with any other non-special key unless Control is also a modifier.

    Parameters
    ----------
    key : str or vispy.app.Key
        Base key.
    modifiers : sequence of str or vispy.app.Key
        Modifier keys.

    Returns
    -------
    seq : str
        Generated key sequence.
    """
    if len(key) == 1 and key.isalpha():  # it's a letter
        if 'Shift' not in modifiers:
            key = key.lower()
            cond = lambda m: True  # noqa: E731
        else:
            # Shift is consumed to create upper-case letter
            key = key.upper()
            cond = lambda m: m != 'Shift'  # noqa: E731
    elif key in SPECIAL_KEYS:
        # remove redundant information i.e. an output of 'Shift-Shift'
        cond = lambda m: m != key  # noqa: E731
    else:
        # Shift is consumed to transform key

        # bug found on OSX: Command will cause Shift to not
        # transform the key so do not consume it
        # note: 'Control' is OSX Command key
        cond = lambda m: m != 'Shift' or 'Control' in modifiers  # noqa: E731

    modifiers = tuple(key.name for key in
                      filter(lambda key: key in modifiers and cond(key),
                             MODIFIER_KEYS.values()))

    return '-'.join(modifiers + (key,))


def normalize_key_sequence(seq):
    """Normalize key sequence to make it easily comparable.

    All aliases are converted and modifier orders are fixed to:
    Control, Alt, Shift, Meta

    Parameters
    ----------
    seq : str
        Key sequence.

    Shift pressed with any letter will be consumed
    to transform it to upper case, e.g. Shift + w => W.
    Otherwise, the letter will be lower case.
    Shift will also be consumed without transformation when pressed
    with any other non-special key unless Control is also a modifier.

    Returns
    -------
    normalized_seq : str
        Normalized key sequence.
    """
    seq = _expand_shorthand(seq)

    key, modifiers = parse_seq(seq)

    if len(key) != 1 and key not in SPECIAL_KEYS:
        raise TypeError(f'invalid key {key}')

    for modifier in modifiers:
        if modifier not in MODIFIER_KEYS.values():
            raise TypeError(f'invalid modifier key {modifier}')

    return components_to_seq(key, modifiers)


def _determine_binding_op(arg):
    """Determine the binding operation based on the argument passed.

    None     -> BindOperation.UNBIND
    str      -> BindOperation.REBIND
    callable -> BindOperation.BIND
    """
    if arg is None:
        return BindOperation.UNBIND
    elif isinstance(arg, str):
        return BindOperation.REBIND
    elif isinstance(arg, Callable):
        return BindOperation.BIND
    raise TypeError('expected arg to be a string, callable, or None; '
                    f'got {type(arg)}')


def _bind_key(keybindings, seq, func):
    """Bind a key sequence to a keymap.
    """
    keybindings[seq] = func


def _unbind_key(keybindings, seq):
    """Unbind a function from a keymap based on its sequence and return it.
    """
    return keybindings.pop(seq)


def _rebind_key(keybindings, new_seq, orig_seq):
    """Rebind a function from one key sequence to another.
    """
    func = _unbind_key(keybindings, orig_seq)
    _bind_key(keybindings, new_seq, func)


def bind_key(keybindings, seq, func):
    """Bind a key sequence to a keymap.

    Parameters
    ----------
    keybindings : dict of str: callable
        Keymap to modify.
    seq : str
        Key sequence.
    func : callable, str, or None
        Callable to bind to the key sequence.
        If a string is passed, instead perform a rebind operation.
        If ``None`` is passed, unbind instead.
    """
    seq = normalize_key_sequence(seq)

    bind_op = _determine_binding_op(func)

    if bind_op == BindOperation.UNBIND:
        try:
            _unbind_key(keybindings, seq)
        except KeyError:
            pass
    elif bind_op == BindOperation.REBIND:
        orig_seq = normalize_key_sequence(func)
        _rebind_key(keybindings, seq, orig_seq)
    else:
        _bind_key(keybindings, seq, func)


def unbind_key(keybindings, seq):
    """Unbind a function from a keymap.

    Parameters
    ----------
    keybindings : dict of str: callable
        Keymap to modify.
    seq : str
        Key sequence.

    Returns
    -------
    func : callable
       Unbound function.

    Raises
    ------
    KeyError
        When the key sequence is not found in the keymap.
    """
    seq = normalize_key_sequence(seq)
    return _unbind_key(keybindings, seq)


def rebind_key(keybindings, new_seq, orig_seq):
    """Rebind a function from one key sequence to another.

    Parameters
    ----------
    keybindings : dict of str: callable
        Keymap to modify.
    new_seq : str
        Key sequence to rebind to.
    orig_seq : str
        Key sequence to unbind from.

    Raises
    ------
    KeyError
        When the original key sequence is not found in the keymap.
    """
    new_seq = normalize_key_sequence(new_seq)
    orig_seq = normalize_key_sequence(orig_seq)

    _rebind_key(keybindings, new_seq, orig_seq)


def _bind_keys_validate_pair(pair):
    """Raise a TypeError if the given pair is not
    a 2-sequence of (str, (callable, str, or None))
    """
    if not (isinstance(pair, Sequence) and not isinstance(pair, STRINGTYPES)):
        raise TypeError(f'expected {pair} to be a sequence, got {type(pair)}')
    key, op = pair
    if not isinstance(key, str):
        raise TypeError(f'key {key} must be a string, not {type(key)}')
    _determine_binding_op(op)


def unbind_keys(keybindings, *seqs, error=True):
    """Unbind functions from a keymap.

    Parameters
    ----------
    keybindings : dict of str: callable
        Keymap to modify.
    *seqs : iterable of str
        Key sequences to unbind. If not provided, unbind all key sequences.
    error : bool
        Whether to error when a function is not found in the keymap.
        Defaults to ``True``.

    Returns
    -------
    funcs : list of callable
        Unbound functions.

    Raises
    ------
    KeyError
        When a function is not found in the keymap
        and ``error`` is set to ``True``.
    """
    if len(seqs) == 0:
        seqs = list(keybindings.keys())
    elif len(seqs) == 1 and isinstance(seqs[0], Iterable):
        seqs = seqs[0]

    funcs = []

    for seq in seqs:
        try:
            funcs.append(unbind_key(keybindings, seq))
        except KeyError:
            if error:
                # rebind unbound functions
                bind_keys(keybindings, zip(seqs[:len(funcs)], funcs))
                raise

    return funcs


def rebind_keys(keybindings, key_pairs):
    """Rebind functions to different key sequences.

    Parameters
    ----------
    keybindings : dict of str: callable
        Keymap to modify.
    key_pairs : iterable of 2-sequence of (str, str)
        Key pairs mapping new key sequences to original key sequences.

    Raises
    ------
    KeyError
        When a function is not found in the keymap.

    Notes
    -----
    First, all unbinding operations are performed, then all binding operations.
    This allows the effective swapping of key sequences between two functions.
    """
    new_seqs = []
    orig_seqs = []

    for new_seq, orig_seq in key_pairs:
        new_seqs.append(new_seq)
        orig_seqs.append(orig_seq)

    funcs = unbind_keys(keybindings, orig_seqs)
    bind_keys(keybindings, zip(new_seqs, funcs))


def bind_keys(keybindings, key_pairs):
    """Bind, unbind, or rebind multiple key sequences at the same time.

    Parameters
    ----------
    keybindings : dict of str: callable
        Keymap to modify.
    key_pairs : iterable of 2-sequence of (str, (callable, str, or None))
        Key pairs mapping key sequences to functions.
        If a string is passed, instead perform a rebind operation.
        If ``None`` is passed, unbind instead.

    Notes
    -----
    Unbindings are performed first, followed by rebindings,
    and finally new bindings.
    """
    to_rebind = []
    to_unbind = []
    to_bind = []

    for pair in key_pairs:
        _bind_keys_validate_pair(pair)
        _, arg = pair
        op = _determine_binding_op(arg)

        if op == BindOperation.REBIND:
            to_rebind.append(pair)
        elif op == BindOperation.UNBIND:
            to_unbind.append(pair[0])
        elif op == BindOperation.BIND:
            to_bind.append(pair)

    if to_unbind:
        unbind_keys(keybindings, to_unbind, error=False)

    if to_rebind:
        rebind_keys(keybindings, to_rebind)

    for seq, func in to_bind:
        bind_key(keybindings, seq, func)


def _bind_key_method_normalize_input(args, kwargs):
    """Normalize args and kwargs to key-operation pairs.

    Parameters
    ----------
    args : 2-tuple of str, (callable, str, or None), \
           1-tuple of dict of str: (callable, str, or None), \
           1-tuple of iterable of 2-sequence of (str, (callable, str, or None)), \  # noqa
           or tuple of 2-sequence of (str, (callable, str, or None))
        Binding operations taken as positional arguments.
    kwargs : dict of str: (callable, str, or None)
        Binding operations taken as keyword arguments.

    Returns
    -------
    ops : tuple of 2-sequence of (str, (callable, str, or None))
        Binding operations in a singular format.
    """
    if len(args) == 1:
        # bind_key((('asdf', func), ('sdf', None)))
        # is equivalent to
        # bind_key(('asdf', func), ('sdf', None))
        args = args[0]

        if isinstance(args, Mapping):
            args = args.items()  # dict -> (N, 2) view

        if isinstance(args, Iterable):
            args = tuple(args)
    elif len(args) == 2:
        # bind_key('asdf', func) == bind_key(('asdf', func))
        try:
            _bind_keys_validate_pair(args)
        except TypeError:
            pass
        else:
            args = (args,)

    args += tuple(kwargs.items())

    return args


def _bind_key_method(keybindings, *args, **kwargs):
    """Bind, unbind, or rebind one or more key sequences at the same time.

    ``o.bind_key(seq, op)``

    ``@o.bind_key(seq)`` (used as a decorator)

    ``o.bind_key({seq1: op1, seq2: op2, ..., seqn: opn})``

    ``o.bind_key((seq1, op1), (seq2, op2), ..., (seqn, opn))``

    ``o.bind_key([(seq1, op1), (seq2, op2), ..., (seqn, opn)])``

    ``o.bind_key(seq1=op1, seq2=op2, ..., seqn=opn)``

    Parameters
    ----------
    operations : dict of str: (callable, str, or None) \
                 or iterable of 2-sequence of (str, (callable, str, or None))
        Key sequence and binding operation pairs.
        Passing a callable will bind it to the key sequence.
        Passing another key sequence  will rebind it to the new key sequence.
        Passing ``None`` will unbind the key sequence.

    Notes
    -----
    Unbindings are performed first, followed by rebindings,
    and finally new bindings.
    """
    if len(args) == 1 and isinstance(args[0], str):
        def inner(func):
            bind_key(keybindings, args[0], func)
            return func

        return inner

    bind_keys(keybindings, _bind_key_method_normalize_input(args, kwargs))


class KeybindingDescriptor:
    """Method descriptor that binds `cls._method` to an instance's
    and optionally a class's keybindings, similarly to how `self` is bound
    in normal methods.

    Parameters
    ----------
    keybindings : str, keyword-only
        Name of the instance attribute of type ``dict of str: callable``
        to use as a keymap.
    class_keybindings : str, keyword-only, optional
        Name of the class attribute of type ``dict of str: callable``
        to use as a keymap.
    """
    def __init__(self, *, keybindings, class_keybindings=None):
        self.keybindings = keybindings
        self.class_keybindings = class_keybindings

    def __get__(self, instance, klass):
        if instance is None:  # used on class
            if self.class_keybindings is None:
                # no class keybindings; behave normally
                return self._method

            keybindings = getattr(klass, self.class_keybindings)
        else:
            keybindings = getattr(instance, self.keybindings)

        return types.MethodType(self._method, keybindings)


def _keybinding_method(name, method):
    return type(name, (KeybindingDescriptor,),
                {'_method': staticmethod(method)})


bind_key_method = _keybinding_method('bind_key', _bind_key_method)
unbind_keys_method = _keybinding_method('unbind_keys', unbind_keys)
