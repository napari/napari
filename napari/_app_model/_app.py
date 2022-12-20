from __future__ import annotations

from functools import lru_cache
from itertools import chain
from typing import Callable, Dict, Set

from app_model import Application
from app_model.types import Action

from napari._app_model._submenus import SUBMENUS
from napari._app_model.actions import RepeatableAction
from napari._app_model.actions._help_actions import HELP_ACTIONS
from napari._app_model.actions._image_actions import IMAGE_ACTIONS
from napari._app_model.actions._layer_actions import LAYER_ACTIONS
from napari._app_model.actions._points_actions import POINTS_ACTIONS
from napari._app_model.actions._view_actions import VIEW_ACTIONS
from napari._app_model.actions._viewer_actions import VIEWER_ACTIONS
from napari._app_model.injection._processors import PROCESSORS
from napari._app_model.injection._providers import PROVIDERS
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Points
from napari.utils.action_manager import action_manager


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

        for action in chain(
            HELP_ACTIONs,
            IMAGE_ACTIONS,
            LAYER_ACTIONS,
            POINTS_ACTIONS,
            VIEW_ACTIONS,
            VIEWER_ACTIONS,
        ):
            self.register_action(action)

        # re-register with action_manager shim for keybindings
        for keymapprovider, actions in (
            (ViewerModel, chain(VIEW_ACTIONS, VIEWER_ACTIONS)),
            (Image, IMAGE_ACTIONS),
            # TODO: (Labels, LABELS_ACTIONS),
            (Points, POINTS_ACTIONS),
            # TODO: (Shapes, SHAPES_ACTIONS),
        ):
            for action in actions:
                self._register_action_manager_shim(action, keymapprovider)

        self.menus.append_menu_items(SUBMENUS)

    def _register_action_manager_shim(self, action: Action, keymapprovider):
        """Shim from app-model Action to action_manager for keybinding"""
        # TODO: remove this once keybind handling is ported to app-model
        # this is a hack because action_manager actions can only be
        # "prefix:suffix", while app-model supports extra levels of nesting
        # e.g.
        #     app-model allows "napari:viewer:toggle_theme"
        #     action_manager wants "napari:toggle_theme"
        # so this hack works if we keep the prefix and suffix the same when
        # porting to app-model
        prefix, *group, command = action.id.split(":")

        def _callback(*args, **kwargs):
            print(f"calling {action.id} via action_manager shim")
            self.get_app().commands.execute_command(action.id).result()

        action_manager.register_action(
            name=f"{prefix}:{command}",
            command=_callback,
            description=action.title,
            keymapprovider=keymapprovider,
            # repeatable=isinstance(action, RepeatableAction),
        )

    @classmethod
    def get_app(cls) -> NapariApplication:
        return Application.get_app(APP_NAME) or cls()

    def register_action(self, action: Action) -> Callable[[], None]:
        dispose = super().register_action(action)
        if isinstance(action, RepeatableAction) and action.repeatable:
            self.set_action_repeatable(action.id, True)

            def _dispose():
                dispose()
                self._repeatable_actions.remove(action.id)

            return _dispose
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
