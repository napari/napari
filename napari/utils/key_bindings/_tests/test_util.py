import pytest
from app_model.types import KeyChord, KeyCode, KeyMod
from app_model.types._constants import OperatingSystem

from napari.utils.key_bindings.util import create_conflict_filter, key2mod


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
