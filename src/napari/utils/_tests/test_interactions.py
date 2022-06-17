import pytest

from ..interactions import Shortcut


@pytest.mark.parametrize(
    'shortcut,reason',
    [
        ('Ctrl-A', 'Ctrl instead of Control'),
        ('Ctrl+A', '+ instead of -'),
        ('Ctrl-AA', 'AA make no sens'),
        ('BB', 'BB make no sens'),
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

    assert Shortcut('Control-A').qt == 'Control+A'
