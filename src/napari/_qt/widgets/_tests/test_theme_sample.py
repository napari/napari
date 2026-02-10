from napari._qt.widgets.qt_theme_sample import SampleWidget, ThemeColorDisplay


def test_theme_sample(qtbot):
    """Just a smoke test to make sure that the theme sample can be created."""
    w = SampleWidget()
    qtbot.addWidget(w)
    w.show()
    assert w.isVisible()


def test_theme_color_display(qtbot):
    """Just a smoke test to make sure that the theme color display can be created."""
    w = ThemeColorDisplay()
    qtbot.addWidget(w)
    w.show()
    assert w.isVisible()
