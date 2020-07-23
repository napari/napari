from napari._qt import theme_sample


def test_theme_sample(qtbot):
    """Just a smoke test to make sure that the theme sample can be created."""
    w = theme_sample.SampleWidget()
    qtbot.addWidget(w)
    w.show()
    assert w.isVisible()
