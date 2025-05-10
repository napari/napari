from __future__ import annotations

from functools import lru_cache

from app_model import Application

from napari._app_model.actions._layerlist_context_actions import (
    LAYERLIST_CONTEXT_ACTIONS,
    LAYERLIST_CONTEXT_SUBMENUS,
)
from napari._app_model.actions._view import VIEW_ACTIONS

APP_NAME = 'napari'


class NapariApplication(Application):
    """A singleton (per name) class representing the Napari application.

    This class extends the app_model.Application class and provides a
    singleton instance of the Napari application. It is responsible for
    managing the application state, including the injection store and
    registering actions and menus.
    """

    def __init__(self, app_name=APP_NAME) -> None:
        # raise_synchronous_exceptions means that commands triggered via
        # ``execute_command`` will immediately raise exceptions. Normally,
        # `execute_command` returns a Future object (which by definition does not
        # raise exceptions until requested).  While we could use that future to raise
        # exceptions with `.result()`, for now, raising immediately should
        # prevent any unexpected silent errors.  We can turn it off later if we
        # adopt asynchronous command execution.
        super().__init__(app_name, raise_synchronous_exceptions=True)

        self.injection_store.namespace = _napari_names  # type: ignore [assignment]

        self.register_actions(LAYERLIST_CONTEXT_ACTIONS)
        self.register_actions(VIEW_ACTIONS)
        self.menus.append_menu_items(LAYERLIST_CONTEXT_SUBMENUS)

    @classmethod
    def get_app_model(cls, app_name: str = APP_NAME) -> NapariApplication:
        """Get the Napari Application singleton.

        This class method that returns the singleton instance of the
        NapariApplication. It relies on the parent class's Application.get_app()
        method (provided by the app_model library) to retrieve the application
        instance by name.
        """
        return Application.get_app(app_name) or cls()


@lru_cache(maxsize=1)
def _napari_names() -> dict[str, object]:
    """Napari names to inject into local namespace when evaluating type hints."""
    import napari
    from napari import components, layers, viewer

    def _public_types(module):
        return {
            name: val
            for name, val in vars(module).items()
            if not name.startswith('_')
            and isinstance(val, type)
            and getattr(val, '__module__', '_').startswith('napari')
        }

    return {
        'napari': napari,
        **_public_types(components),
        **_public_types(layers),
        **_public_types(viewer),
    }


def get_app_model() -> NapariApplication:
    """Get the Napari Application singleton.

    This public function returns the singleton instance of the
    NapariApplication.
    """
    return NapariApplication.get_app_model()
