import sys

import pytest

from napari.utils.tips import format_tip


@pytest.mark.parametrize(
    ('raw', 'formatted_linux', 'formatted_win', 'formatted_mac'),
    [
        (
            'press {Meta+K} to jump',
            'press Super+K to jump',
            'press ⊞+K to jump',
            'press ⌘K to jump',
        ),
        ('{Space} or {Alt}', '␣ or Alt', '␣ or Alt', '␣ or ⌥'),  # multiple
        ('{napari.viewer.fit_to_view}', 'Ctrl+0', 'Ctrl+0', '⌘0'),  # appmodel
        ('{napari:reset_view}', 'Ctrl+R', 'Ctrl+R', '⌘R'),  # action manager
    ],
)
def test_format_tips(raw, formatted_linux, formatted_win, formatted_mac):
    match sys.platform:
        case 'linux':
            expected = formatted_linux
        case 'win32':
            expected = formatted_win
        case 'darwin':
            expected = formatted_mac

    assert format_tip(raw) == expected
