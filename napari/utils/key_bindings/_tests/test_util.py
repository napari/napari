import pytest
from app_model.types import KeyBinding, KeyChord, KeyCode, KeyMod
from app_model.types._constants import OperatingSystem

from napari.utils.key_bindings.util import (
    create_conflict_filter,
    key2mod,
    validate_key_binding,
)


def test_key2mod():
    # win
    assert key2mod(KeyCode.Alt, OperatingSystem.WINDOWS) == KeyMod.Alt
    assert key2mod(KeyCode.Shift, OperatingSystem.WINDOWS) == KeyMod.Shift
    assert key2mod(KeyCode.Ctrl, OperatingSystem.WINDOWS) == KeyMod.CtrlCmd
    assert key2mod(KeyCode.Meta, OperatingSystem.WINDOWS) == KeyMod.WinCtrl
    assert key2mod(KeyCode.KeyA, OperatingSystem.WINDOWS) is None

    # mac
    assert key2mod(KeyCode.Alt, OperatingSystem.MACOS) == KeyMod.Alt
    assert key2mod(KeyCode.Shift, OperatingSystem.MACOS) == KeyMod.Shift
    assert key2mod(KeyCode.Ctrl, OperatingSystem.MACOS) == KeyMod.WinCtrl
    assert key2mod(KeyCode.Meta, OperatingSystem.MACOS) == KeyMod.CtrlCmd
    assert key2mod(KeyCode.KeyA, OperatingSystem.MACOS) is None

    # linux
    assert key2mod(KeyCode.Alt, OperatingSystem.LINUX) == KeyMod.Alt
    assert key2mod(KeyCode.Shift, OperatingSystem.LINUX) == KeyMod.Shift
    assert key2mod(KeyCode.Ctrl, OperatingSystem.LINUX) == KeyMod.CtrlCmd
    assert key2mod(KeyCode.Meta, OperatingSystem.LINUX) == KeyMod.WinCtrl
    assert key2mod(KeyCode.KeyA, OperatingSystem.LINUX) is None


def test_create_conflict_filter():
    key_seqs = [
        KeyCode.KeyA,
        KeyChord(KeyCode.KeyA, KeyCode.KeyB),
        KeyMod.Shift | KeyCode.KeyA,
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyB,
        KeyMod.Alt,
        KeyChord(
            KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC,
            KeyMod.Alt | KeyCode.KeyD,
        ),
    ]

    # should catch a chord of `a b`, but not `shift+a`
    a_fltr = create_conflict_filter(KeyCode.KeyA)
    assert set(filter(a_fltr, key_seqs)) == {
        KeyChord(KeyCode.KeyA, KeyCode.KeyB)
    }

    # should NOT catch chord `a b` or `ctrl+shift+b`
    b_fltr = create_conflict_filter(KeyCode.KeyB)
    assert set(filter(b_fltr, key_seqs)) == set()

    # should catch all entries with shift in the first part
    shift_fltr = create_conflict_filter(KeyMod.Shift)
    assert set(filter(shift_fltr, key_seqs)) == {
        KeyMod.Shift | KeyCode.KeyA,
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyB,
        KeyChord(
            KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC,
            KeyMod.Alt | KeyCode.KeyD,
        ),
    }

    # should NOT catch alt in the second part of the chord `ctrl+shift+c alt+d`
    alt_fltr = create_conflict_filter(KeyMod.Alt)
    assert set(filter(alt_fltr, key_seqs)) == set()

    # should be able to combine mods
    ctrl_shift_fltr = create_conflict_filter(KeyMod.CtrlCmd | KeyMod.Shift)
    assert set(filter(ctrl_shift_fltr, key_seqs)) == {
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyB,
        KeyChord(
            KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC,
            KeyMod.Alt | KeyCode.KeyD,
        ),
    }

    # should be able to check for entire first part of a key chord
    chord_fltr = create_conflict_filter(
        KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC
    )
    assert set(filter(chord_fltr, key_seqs)) == {
        KeyChord(
            KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyC,
            KeyMod.Alt | KeyCode.KeyD,
        ),
    }

    # test exception
    with pytest.raises(TypeError):
        create_conflict_filter(KeyChord(KeyCode.KeyA, KeyCode.KeyB))


def test_validate_key_binding():
    kb = KeyBinding.from_str
    kbi = KeyBinding.from_int

    assert validate_key_binding(kb('a b')) == kb('a b')

    with pytest.raises(TypeError) as e:
        validate_key_binding(kbi(KeyCode.IntlBackslash))
    assert e.match('invalid base key')

    with pytest.raises(TypeError) as e:
        validate_key_binding(kb('shift a'))
    assert e.match(
        'cannot have single modifier as base key in a two-part chord'
    )

    with pytest.raises(TypeError) as e:
        validate_key_binding(kb('shift+alt'))
    assert e.match('key combination cannot be comprised of only modifier keys')

    with pytest.raises(TypeError) as e:
        validate_key_binding(
            kbi(KeyChord(KeyCode.KeyA, KeyCode.IntlBackslash))
        )
    assert e.match('invalid base key')

    with pytest.warns(UserWarning):
        assert validate_key_binding(kb('shift+shift')) == kbi(KeyMod.Shift)

    with pytest.warns(UserWarning):
        assert validate_key_binding(kbi(KeyCode.Alt)) == kbi(KeyMod.Alt)

    assert validate_key_binding(kbi(KeyMod.Alt)) == kbi(KeyMod.Alt)

    with pytest.raises(TypeError) as e:
        validate_key_binding(kbi(KeyMod.Alt | KeyMod.Shift))
    assert e.match('must have exactly one modifier as a standalone base key')

    assert validate_key_binding(kb('ctrl+a ctrl+v')) == kb('ctrl+a ctrl+v')

    with pytest.raises(TypeError) as e:
        validate_key_binding(kb('ctrl+a ctrl'))
    assert e.match(
        'cannot have single modifier as base key in a two-part chord'
    )
