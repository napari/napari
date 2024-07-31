from __future__ import annotations

from functools import lru_cache

from app_model import Application

from napari._app_model.actions._layerlist_context_actions import (
    LAYERLIST_CONTEXT_ACTIONS,
    LAYERLIST_CONTEXT_SUBMENUS,
)

APP_NAME = 'napari'


class NapariApplication(Application):
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
        self.menus.append_menu_items(LAYERLIST_CONTEXT_SUBMENUS)

    @classmethod
    def get_app(cls, app_name: str = APP_NAME) -> NapariApplication:
        return Application.get_app(app_name) or cls()  # type: ignore[return-value]


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


def get_app() -> NapariApplication:
    """Get the Napari Application singleton."""
    return NapariApplication.get_app()
