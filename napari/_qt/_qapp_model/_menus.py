from typing import TYPE_CHECKING, Optional

from app_model.backends.qt import QModelMenu

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


def build_qmodel_menu(
    menu_id: str,
    title: Optional[str] = None,
    parent: Optional['QWidget'] = None,
) -> QModelMenu:
    """Build a QModelMenu from the napari app model
    Parameters
    ----------
    menu_id : str
        ID of a menu registered with napari._app_model.get_app().menus
    title : Optional[str]
        Title of the menu
    parent : Optional[QWidget]
        Parent of the menu

    Returns
    -------
    QModelMenu
        QMenu subclass populated with all items in `menu_id` menu.
    """
    from ..._app_model import get_app

    return QModelMenu(
        menu_id=menu_id, app=get_app(), title=title, parent=parent
    )
