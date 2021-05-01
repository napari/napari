import pytest

from ..interactions import ReadOnlyWrapper, Shortcut


def test_ReadOnlyWrapper_setitem():
    """test that ReadOnlyWrapper prevents setting items"""
    d = {'hi': 3}
    d_read_only = ReadOnlyWrapper(d)

    with pytest.raises(TypeError):
        d_read_only['hi'] = 5


def test_ReadOnlyWrapper_setattr():
    """test that ReadOnlyWrapper prevents setting attributes"""

    class TestClass:
        x = 3

    tc = TestClass()
    tc_read_only = ReadOnlyWrapper(tc)

    with pytest.raises(TypeError):
        tc_read_only.x = 5


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


def test_shortcut_qt():

    assert Shortcut('Control-A').qt == 'Control+A'
