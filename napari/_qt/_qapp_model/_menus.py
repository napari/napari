from app_model.backends.qt import QModelMenu


def build_qmodel_menu(menu_id: str) -> QModelMenu:
    """Build a QModelMenu from the napari app model

    Parameters
    ----------
    menu_id : str
        ID of a menu registered with napari._app_model.get_app().menus

    Returns
    -------
    QModelMenu
        QMenu subclass populated with all items in `menu_id` menu.
    """
    from ..._app_model import get_app

    return QModelMenu(menu_id=menu_id, app=get_app())
