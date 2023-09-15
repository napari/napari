import warnings
from typing import Callable, Optional

from app_model.types import KeyBinding, KeyCode, KeyMod
from app_model.types._constants import OperatingSystem

from napari.utils.key_bindings.constants import (
    KEY_MOD_MASK,
    PART_0_MASK,
    VALID_KEYS,
)


def key2mod(key: KeyCode, os: OperatingSystem) -> Optional[KeyMod]:
    """Convert a KeyCode into its KeyMod equivalent.

    Parameters
    ----------
    key : KeyCode
        Key to convert.
    os : OperatingSystem
        Operating system to determine transformation.

    Returns
    -------
    mod : KeyMod or None
        KeyMod equivalent of the given KeyCode if it exists.
    """
    if key == KeyCode.Shift:
        return KeyMod.Shift
    if key == KeyCode.Alt:
        return KeyMod.Alt
    if key == KeyCode.Ctrl:
        if os == OperatingSystem.MACOS:
            return KeyMod.WinCtrl
        return KeyMod.CtrlCmd
    if key == KeyCode.Meta:
        if os == OperatingSystem.MACOS:
            return KeyMod.CtrlCmd
        return KeyMod.WinCtrl
    return None


def create_conflict_filter(conflict_key: int) -> Callable[[int], bool]:
    """Generate a filter function that detects all key sequences
    starting with that sub-sequence, excluding itself.

    Parameters
    ----------
    conflict_key : int
        16-bit key sequence which may be represented by a KeyCode, KeyMod, or KeyCombo.

    Raises
    ------
    TypeError
        When the given conflict key sequence is not 16 bits or less

    Returns
    -------
    filter_func : Callable[[int], bool]
        Generated filter function.
    """
    if conflict_key > PART_0_MASK:
        # don't handle anything more complex
        raise TypeError(
            f'filter creation only works on one-part key sequences (16-bit integers), not {conflict_key}'
        )

    if conflict_key & KEY_MOD_MASK == conflict_key:
        # only comprised of modifier keys in first part
        def inner(key: int) -> bool:
            return key != conflict_key and key & conflict_key == conflict_key

    else:
        # one-part key sequence
        def inner(key: int) -> bool:
            return key > PART_0_MASK and key & PART_0_MASK == conflict_key

    return inner


def validate_key_binding(key: KeyBinding, warn: bool = True) -> KeyBinding:
    """Check if the given key binding matches the criteria for a valid key binding and normalizes it as needed.

    Parameters
    ----------
    key : KeyBinding
        Key binding to validate.
    warn : bool, optional
        Whether to raise a warning when a single modifier binding has a modifier as the base key.

    Raises
    ------
    TypeError
        When the entered key binding is invalid.

    Warns
    -----
    UserWarning
        When single modifier binding has modifier as the base key.

    Returns
    -------
    normalized_key: KeyBinding
        Validated and normalized key binding.
    """
    n_parts = len(key.parts)
    if n_parts > 2:
        raise TypeError('cannot have more than two parts')
    if n_parts == 0:
        raise TypeError('must have at least one part')

    if key.part0.is_modifier_key():
        if n_parts == 2:
            raise TypeError(
                'cannot have single modifier as base key in a two-part chord'
            )
        n_mods = sum(
            (key.part0.ctrl, key.part0.shift, key.part0.alt, key.part0.meta)
        )
        if key.part0.key == KeyCode.UNKNOWN:
            if n_mods != 1:
                raise TypeError(
                    'must have exactly one modifier as a standalone base key'
                )
        else:
            base_key = str(key.part0.key).lower()
            if n_mods > 1 or (
                n_mods == 1 and not getattr(key.part0, base_key)
            ):
                raise TypeError(
                    'key combination cannot be comprised of only modifier keys'
                )
            if warn:
                warnings.warn(
                    f"using '{base_key}' as base key; use as a modifier instead",
                    UserWarning,
                    stacklevel=2,
                )
            return KeyBinding.from_int(
                key2mod(key.part0.key, OperatingSystem.current())
            )
    else:
        if key.part0.key not in VALID_KEYS:
            raise TypeError(f'invalid base key {key.part0.key}')

    if n_parts == 2:
        if key.parts[1].is_modifier_key():
            raise TypeError(
                'cannot have single modifier as base key in a two-part chord'
            )

        if key.parts[1].key not in VALID_KEYS:
            raise TypeError(f'invalid base key {key.parts[1].key}')

    return key


def kb2str(kb: KeyBinding) -> str:
    result = str(kb)
    # when only modifier keys present the default implementation will add a + at the end
    if result.endswith('+'):
        result = result[:-1]
    return result.lower()
