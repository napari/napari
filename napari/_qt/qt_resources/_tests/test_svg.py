from qtpy.QtGui import QIcon

from napari._qt.qt_resources import QColoredSVGIcon


def test_colored_svg(qtbot):
    """Test that we can create a colored icon with certain color."""
    icon = QColoredSVGIcon.from_resources('new_points')
    assert isinstance(icon, QIcon)
    assert isinstance(icon.colored('#0934e2', opacity=0.4), QColoredSVGIcon)
    assert icon.pixmap(250, 250)


def test_colored_svg_from_theme(qtbot):
    """Test that we can create a colored icon using a theme name."""
    icon = QColoredSVGIcon.from_resources('new_points')
    assert isinstance(icon, QIcon)
    assert isinstance(icon.colored(theme='light'), QColoredSVGIcon)
    assert icon.pixmap(250, 250)


def test_colored_svg_cache(qtbot):
    """Make sure we're not recreating icons."""
    icon1 = QColoredSVGIcon.from_resources('new_points')
    icon2 = QColoredSVGIcon.from_resources('new_points')
    assert icon1 is icon2
    assert icon1.colored('red') is icon2.colored('red')
