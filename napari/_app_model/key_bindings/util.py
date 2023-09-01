from typing import Callable, Optional

from app_model.types import KeyCode, KeyMod
from app_model.types._constants import OperatingSystem

from napari._app_model.key_bindings.constants import KEY_MOD_MASK, PART_0_MASK


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
        if os.is_mac:
            return KeyMod.WinCtrl
        return KeyMod.CtrlCmd
    if key == KeyCode.Meta:
        if os.is_mac:
            return KeyMod.CtrlCmd
        return KeyMod.WinCtrl
    return None


def create_conflict_filter(conflict_key: int) -> Callable[[int], bool]:
    """Generate a filter function that detects all key sequences
    starting with that sub-sequence, excluding itself.

    Parameters
    ----------
    conflict_key : int
        32-bit key sequence which may be
        represented by a KeyCode, KeyMod, KeyCombo, or KeyChord.

    Returns
    -------
    filter_func : Callable[[int], bool]
        Generated filter function.
    """
    if conflict_key & KEY_MOD_MASK == conflict_key:
        # only comprised of modifier keys in first part
        def inner(key: int) -> bool:
            return key != conflict_key and key & conflict_key

    elif conflict_key <= PART_0_MASK:
        # one-part key sequence
        def inner(key: int) -> bool:
            return key > PART_0_MASK and key & PART_0_MASK == conflict_key

    else:
        # don't handle anything more complex
        def inner(key: int) -> bool:
            return NotImplemented

    return inner
