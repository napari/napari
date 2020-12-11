import pytest

from napari.components.style import Style


def test_changing_theme():
    """Test changing the theme."""
    style = Style()
    assert style.theme == 'dark'
    assert style.palette['folder'] == 'dark'

    style.theme = 'light'
    assert style.palette['folder'] == 'light'


def test_non_existant_theme():
    """Test non existant theme raises an error."""
    style = Style()
    assert style.theme == 'dark'

    with pytest.raises(ValueError):
        style.theme = 'nonexistent_theme'
