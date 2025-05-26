import qtpy
from qtpy.QtWidgets import QMenu


def get_submenu_action(qmodel_menu, submenu_text, action_text):
    """
    Get an action that belongs to a submenu inside a `QModelMenu` instance.

    Needed since `QModelMenu.findAction` will not search inside submenu actions

    Parameters
    ----------
    qmodel_menu : app_model.backends.qt.QModelMenu
        One of the application menus created via `napari._qt._qapp_model.build_qmodel_menu`.
    submenu_text : str
        Text of the submenu where an action should be searched.
    action_text : str
        Text of the action to search for.

    Raises
    ------
    ValueError
        In case no action could be found.

    Returns
    -------
    tuple[QAction, QAction]
        Tuple with submenu action and found action.

    """

    def _get_menu(act):
        # this function may be removed when PyQt6 will release next version
        # (after 6.3.1 - if we do not want to support this test on older PyQt6)
        # https://www.riverbankcomputing.com/pipermail/pyqt/2022-July/044817.html
        # because both PyQt6 and PySide6 will have working menu method of action
        return (
            QMenu.menuInAction(act)
            if getattr(qtpy, 'PYQT6', False)
            else act.menu()
        )

    actions = qmodel_menu.actions()
    for action1 in actions:
        if action1.text() == submenu_text:
            for action2 in _get_menu(action1).actions():
                if action2.text() == action_text:
                    return action2, action1
    raise ValueError(
        f'Could not find action "{action_text}" in "{submenu_text}"'
    )  # pragma: no cover
