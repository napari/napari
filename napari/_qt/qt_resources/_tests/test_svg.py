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
