import pytest

from napari.utils.interactions import Shortcut


@pytest.mark.parametrize(
    'shortcut,reason',
    [
        ('Atl-A', 'Alt misspelled'),
        ('Ctrl-AA', 'AA makes no sense'),
        ('BB', 'BB makes no sense'),
    ],
)
def test_shortcut_invalid(shortcut, reason):
    with pytest.warns(UserWarning):
        Shortcut(shortcut)  # Should be Control-A


def test_minus_shortcut():
    """
    Misc tests minus is properly handled as it is the delimiter
    """
    assert str(Shortcut('-')) == '-'
    assert str(Shortcut('Control--')).endswith('-')
    assert str(Shortcut('Shift--')).endswith('-')


def test_shortcut_qt():
    assert Shortcut('Control-A').qt == 'Ctrl+A'
