def test_resources():
    from . import qt

    assert qt.QtCore.__package__ == 'qtpy'
