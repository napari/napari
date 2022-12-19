from __future__ import annotations

from functools import lru_cache
from itertools import chain
from typing import Callable, Dict, Set

from app_model import Application
from app_model.types import Action

from napari._app_model._submenus import SUBMENUS
from napari._app_model.actions import RepeatableAction
from napari._app_model.actions._help_actions import HELP_ACTIONS
from napari._app_model.actions._layer_actions import LAYER_ACTIONS
from napari._app_model.actions._view_actions import VIEW_ACTIONS
from napari._app_model.injection._processors import PROCESSORS
from napari._app_model.injection._providers import PROVIDERS

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

        self._repeatable_actions: Set[str] = set()

        self.injection_store.namespace = _napari_names  # type: ignore [assignment]
        self.injection_store.register(
            providers=PROVIDERS, processors=PROCESSORS
        )

        for action in chain(HELP_ACTIONS, LAYER_ACTIONS, VIEW_ACTIONS):
            self.register_action(action)

        self.menus.append_menu_items(SUBMENUS)

    @classmethod
    def get_app(cls) -> NapariApplication:
        return Application.get_app(APP_NAME) or cls()

    def register_action(self, action: Action) -> Callable[[], None]:
        dispose = super().register_action(action)
        if isinstance(action, RepeatableAction) and action.repeatable:
            self.set_action_repeatable(action.id, True)
        return dispose

    def set_action_repeatable(self, action_id: str, repeatable: bool):
        """Registers an action as repeatable or not.

        Parameters
        ----------
        action_id : str
            Unique identifier of the action.
        repeatable : bool
            Whether or not to set the action as repeatable.
        """
        if action_id not in self.commands:
            raise ValueError(f'Command {id!r} not registered')

        if repeatable:
            self._repeatable_actions.add(action_id)
        elif self.action_is_repeatable(action_id):
            self._repeatable_actions.remove(action_id)

    def action_is_repeatable(self, action_id: str) -> bool:
        """Determines if an action is repeatable.

        Parameters
        ----------
        action_id : str
            Unique identifier of the action.

        Returns
        -------
        bool
            Whether the action is repeatable or not.
        """
        return action_id in self._repeatable_actions


@lru_cache(maxsize=1)
def _napari_names() -> Dict[str, object]:
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
