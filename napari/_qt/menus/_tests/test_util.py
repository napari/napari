from unittest.mock import MagicMock

from qtpy.QtWidgets import QMenu

from napari._qt.menus._util import populate_menu


def test_populate_menu_create(qtbot):
    """Test the populate_menu function."""

    mock = MagicMock()
    menu = QMenu()
    populate_menu(menu, [{"text": "test", "slot": mock}])
    assert len(menu.actions()) == 1
    assert menu.actions()[0].text() == "test"
    assert menu.actions()[0].isCheckable() is False
    with qtbot.waitSignal(menu.actions()[0].triggered):
        menu.actions()[0].trigger()
    mock.assert_called_once()


def test_populate_menu_create_checkable(qtbot):
    """Test the populate_menu function with checkable actions."""

    mock = MagicMock()
    menu = QMenu()
    populate_menu(menu, [{"text": "test", "slot": mock, "checkable": True}])
    assert len(menu.actions()) == 1
    assert menu.actions()[0].text() == "test"
    assert menu.actions()[0].isCheckable() is True
    with qtbot.waitSignal(menu.actions()[0].triggered):
        menu.actions()[0].trigger()
    mock.assert_called_once_with(True)
    mock.reset_mock()
    with qtbot.waitSignal(menu.actions()[0].triggered):
        menu.actions()[0].trigger()
    mock.assert_called_once_with(False)
