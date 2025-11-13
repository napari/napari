from __future__ import annotations

from functools import lru_cache
from itertools import chain

from app_model import Application
from app_model.registries import KeyBindingsRegistry
from app_model.types import KeyBinding, KeyBindingSource

from napari._app_model.actions._file import FILE_ACTIONS, FILE_SUBMENUS
from napari._app_model.actions._layerlist_context_actions import (
    LAYERLIST_CONTEXT_ACTIONS,
    LAYERLIST_CONTEXT_SUBMENUS,
)
from napari._app_model.actions._view import VIEW_ACTIONS, VIEW_SUBMENUS

APP_NAME = 'napari'


class NapariKeyBindingsRegistry(KeyBindingsRegistry):
    """A custom KeyBindingsRegistry for Napari.

    This class extends the KeyBindingsRegistry from the app_model library
    and is tailored for the Napari application. It can be used to manage
    keybindings specific to Napari.
    """

    def get_default_shortcuts(self, command_id: str) -> list[KeyBinding]:
        """Get the default shortcuts for a given command ID.

        Parameters
        ----------
        command_id : str
            The command ID for which to retrieve default shortcuts.

        Returns
        -------
        list[KeyBinding]
            A list of KeyBinding objects representing the default shortcuts
            for the specified command ID.
        """
        return [
            kb.keybinding
            for kb in self._keybindings
            if kb.command_id == command_id
            and kb.source == KeyBindingSource.APP
        ]


class NapariApplication(Application):
    """A singleton (per name) class representing the Napari application.

    This class extends the app_model.Application class and provides a
    singleton instance of the Napari application. It is responsible for
    managing the application state, including the injection store and
    registering actions and menus.
    """

    keybindings: NapariKeyBindingsRegistry

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
            keybindings_reg_class=NapariKeyBindingsRegistry,
        )

        self.injection_store.namespace = _napari_names  # type: ignore [assignment]

        self.register_actions(LAYERLIST_CONTEXT_ACTIONS)
        self.register_actions(VIEW_ACTIONS)
        self.register_actions(FILE_ACTIONS)
        self.menus.append_menu_items(
            chain(LAYERLIST_CONTEXT_SUBMENUS, VIEW_SUBMENUS, FILE_SUBMENUS)
        )

    @classmethod
    def get_app_model(cls, app_name: str = APP_NAME) -> NapariApplication:
        """Get the Napari Application singleton.

        This class method that returns the singleton instance of the
        NapariApplication. It relies on the parent class's Application.get_app()
        method (provided by the app_model library) to retrieve the application
        instance by name.
        """
        return Application.get_app(app_name) or cls()

    def get_app_default_shortcuts(
        self,
    ) -> dict[str, list[KeyBinding]]:
        app_action = [
            x
            for x in self._registered_actions
            if x.startswith(f'{APP_NAME}.') and not x.endswith('dummy')
        ]

        return {
            action: self.keybindings.get_default_shortcuts(action)
            for action in app_action
        }

    def get_app_default_shortcuts_groups(
        self,
    ) -> dict[str, list[str]]:
        """Get the Napari Application default shortcut groups.

        Returns
        -------
        dict[str, list[str]]
            A dictionary where keys are group names and values are lists of
            action names belonging to each group.
        """
        return {'Viewer': list(self.get_app_default_shortcuts()), 'Image': []}


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
