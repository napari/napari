from collections.abc import Collection
from typing import TYPE_CHECKING, Optional

from app_model.backends.qt import QModelToolBar

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


def build_qmodel_toolbar(
    menu_id: str,
    exclude: Optional[Collection[str]] = None,
    title: Optional[str] = None,
    parent: Optional['QWidget'] = None,
) -> QModelToolBar:
    """Build a QModelToolbar from the napari app model

    Parameters
    ----------
    menu_id : str
        ID of a menu registered with napari._app_model.get_app().menus
    exclude: Collection[str] | None
        Optional list of menu ids to exclude from the toolbar, by default None
    title : Optional[str]
        Title of the menu
    parent : Optional[QWidget]
        Parent of the menu

    Returns
    -------
    QModelToolBar
        QToolbar subclass populated with all items in `menu_id` menu except for
        the ones that should be `excluded`.
    """
    from napari._app_model import get_app

    return QModelToolBar(
        menu_id=menu_id,
        app=get_app(),
        exclude=exclude,
        title=title,
        parent=parent,
    )
