from enum import IntEnum, IntFlag, auto
from typing import List

from app_model.types import KeyCode


class DispatchFlags(IntFlag):
    RESET = 0
    DELAY = auto()
    SINGLE_MOD = auto()
    ON_RELEASE = auto()
    TWO_PART = auto()


class KeyBindingWeights(IntEnum):
    CORE = 0
    PLUGIN = 300
    USER = 500


KEY_MOD_MASK = 0x00000F00
PART_0_MASK = 0x0000FFFF

VALID_KEYS: List[KeyCode] = [
    # a-z
    KeyCode.KeyA,
    KeyCode.KeyB,
    KeyCode.KeyC,
    KeyCode.KeyD,
    KeyCode.KeyE,
    KeyCode.KeyF,
    KeyCode.KeyG,
    KeyCode.KeyH,
    KeyCode.KeyI,
    KeyCode.KeyJ,
    KeyCode.KeyK,
    KeyCode.KeyL,
    KeyCode.KeyM,
    KeyCode.KeyN,
    KeyCode.KeyO,
    KeyCode.KeyP,
    KeyCode.KeyQ,
    KeyCode.KeyR,
    KeyCode.KeyS,
    KeyCode.KeyT,
    KeyCode.KeyU,
    KeyCode.KeyV,
    KeyCode.KeyW,
    KeyCode.KeyX,
    KeyCode.KeyY,
    KeyCode.KeyZ,
    # 0-9
    KeyCode.Digit0,
    KeyCode.Digit1,
    KeyCode.Digit2,
    KeyCode.Digit3,
    KeyCode.Digit4,
    KeyCode.Digit5,
    KeyCode.Digit6,
    KeyCode.Digit7,
    KeyCode.Digit8,
    KeyCode.Digit9,
    # fn keys
    KeyCode.F1,
    KeyCode.F2,
    KeyCode.F3,
    KeyCode.F4,
    KeyCode.F5,
    KeyCode.F6,
    KeyCode.F7,
    KeyCode.F8,
    KeyCode.F9,
    KeyCode.F10,
    KeyCode.F11,
    KeyCode.F12,
    # symbols
    KeyCode.Backquote,
    KeyCode.Minus,
    KeyCode.Equal,
    KeyCode.BracketLeft,
    KeyCode.BracketRight,
    KeyCode.Backslash,
    KeyCode.Semicolon,
    KeyCode.Quote,
    KeyCode.Comma,
    KeyCode.Period,
    KeyCode.Slash,
    # navigation
    KeyCode.LeftArrow,
    KeyCode.UpArrow,
    KeyCode.RightArrow,
    KeyCode.DownArrow,
    KeyCode.PageUp,
    KeyCode.PageDown,
    KeyCode.End,
    KeyCode.Home,
    # input
    KeyCode.Tab,
    KeyCode.Enter,
    KeyCode.Escape,
    KeyCode.Space,
    KeyCode.Backspace,
    KeyCode.Delete,
    # special function keys
    KeyCode.PauseBreak,
    KeyCode.CapsLock,
    KeyCode.Insert,
    KeyCode.NumLock,
    KeyCode.PrintScreen,
    # numpad
    KeyCode.Numpad0,
    KeyCode.Numpad1,
    KeyCode.Numpad2,
    KeyCode.Numpad3,
    KeyCode.Numpad4,
    KeyCode.Numpad5,
    KeyCode.Numpad6,
    KeyCode.Numpad7,
    KeyCode.Numpad8,
    KeyCode.Numpad9,
    KeyCode.NumpadDecimal,
    KeyCode.NumpadMultiply,
    KeyCode.NumpadDivide,
    KeyCode.NumpadAdd,
    KeyCode.NumpadSubtract,
]
