from functools import lru_cache

from app_model import Application

from ._menus import SUBMENUS
from .actions._layer_actions import LAYER_ACTIONS


@lru_cache
def get_app() -> Application:
    app = Application(name="napari")
    app.menus.append_menu_items(SUBMENUS)

    for action in LAYER_ACTIONS:
        app.register_action(action)

    return app


app = get_app()
