import sys

import pytest

from napari.utils.tips import format_tip


@pytest.mark.parametrize(
    ('raw', 'formatted', 'formatted_mac'),
    [
        (
            'press {Meta+K} to jump',
            'press Super+K to jump',
            'press ⌘+K to jump',
        ),
        ('{Space} or {Alt}', '␣ or Alt', 'Space or Alt'),  # multiple
        ('{napari.viewer.fit_to_view}', 'Ctrl+0', '⌘+0'),  # appmodel
        ('{napari:reset_view}', 'Ctrl+R', '⌘+R'),  # action manager
    ],
)
def test_format_tips(raw, formatted, formatted_mac):
    expected = formatted_mac if sys.platform == 'darwin' else formatted

    assert format_tip(raw) == expected
