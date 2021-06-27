from napari._qt.widgets.qt_theme_sample import SampleWidget


def test_theme_sample(qtbot):
    """Just a smoke test to make sure that the theme sample can be created."""
    w = SampleWidget()
    qtbot.addWidget(w)
    w.show()
    assert w.isVisible()
