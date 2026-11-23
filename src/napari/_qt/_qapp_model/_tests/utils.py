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

    actions = qmodel_menu.actions()
    for action1 in actions:
        if action1.text() == submenu_text:
            for action2 in action1.menu().actions():
                if action2.text() == action_text:
                    return action2, action1
    raise ValueError(
        f'Could not find action "{action_text}" in "{submenu_text}"'
    )  # pragma: no cover
