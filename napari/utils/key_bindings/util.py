from typing import Callable, Optional

from app_model.types import KeyCode, KeyMod
from app_model.types._constants import OperatingSystem

from napari.utils.key_bindings.constants import KEY_MOD_MASK, PART_0_MASK


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
