from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING

from app_model import Application
from in_n_out import Store

from napari._app_model.actions._file import FILE_ACTIONS, FILE_SUBMENUS
from napari._app_model.actions._layerlist_context_actions import (
    LAYERLIST_CONTEXT_ACTIONS,
    LAYERLIST_CONTEXT_SUBMENUS,
)
from napari._app_model.actions._view import VIEW_ACTIONS, VIEW_SUBMENUS

if TYPE_CHECKING:
    from collections.abc import Generator

APP_NAME = 'napari'


class NapariStore(Store):
    """A store of a singleton class which represents the Napari application with temporary namespace overrides."""

    @contextmanager
    def _add_to_namespace(
        self, name: str, value: object
    ) -> Generator[None, None, None]:
        namespace = self.namespace.copy()
        self.namespace = {**namespace, name: value}
        try:
            yield
        finally:
            self.namespace = namespace


class NapariApplication(Application):
    """A singleton (per name) class representing the Napari application.

    This class extends the app_model.Application class and provides a
    singleton instance of the Napari application. It is responsible for
    managing the application state, including the injection store and
    registering actions and menus.
    """

    injection_store: NapariStore

    def __init__(self, app_name=APP_NAME) -> None:
        # raise_synchronous_exceptions means that commands triggered via
        # ``execute_command`` will immediately raise exceptions. Normally,
        # `execute_command` returns a Future object (which by definition does not
        # raise exceptions until requested).  While we could use that future to raise
        # exceptions with `.result()`, for now, raising immediately should
        # prevent any unexpected silent errors.  We can turn it off later if we
        # adopt asynchronous command execution.
        super().__init__(
            app_name,
            raise_synchronous_exceptions=True,
            injection_store_class=NapariStore,
        )

        self.injection_store.namespace = _napari_names  # type: ignore [assignment]

        self.register_actions(LAYERLIST_CONTEXT_ACTIONS)
        self.register_actions(VIEW_ACTIONS)
        self.register_actions(FILE_ACTIONS)
        self.menus.append_menu_items(
            chain(LAYERLIST_CONTEXT_SUBMENUS, VIEW_SUBMENUS, FILE_SUBMENUS)
        )

    @contextmanager
    def register_with_namespace(self, name: str, obj: object):
        def provider() -> object:
            return obj

        with (
            self.injection_store._add_to_namespace(name, obj.__class__),
            self.injection_store.register(
                providers=[(provider, obj.__class__)]
            ),
        ):
            yield

    @classmethod
    def get_app_model(cls, app_name: str = APP_NAME) -> NapariApplication:
        """Get the Napari Application singleton.

        This class method that returns the singleton instance of the
        NapariApplication. It relies on the parent class's Application.get_app()
        method (provided by the app_model library) to retrieve the application
        instance by name.
        """
        app = Application.get_app(app_name)
        if app is None:
            return cls()
        if not isinstance(app, NapariApplication):
            raise TypeError(
                f'Application `{app_name}` is not a NapariApplication'
            )
        return app


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
