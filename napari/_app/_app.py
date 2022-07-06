from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, Optional, TypeVar

import in_n_out as ino
from app_model import Application

from ._menus import SUBMENUS
from .actions._layer_actions import LAYER_ACTIONS
from .injection._processors import _init_processors
from .injection._providers import _init_providers

if TYPE_CHECKING:
    from in_n_out._type_resolution import RaiseWarnReturnIgnore


C = TypeVar("C", bound=Callable)


class NapariApplication(Application):
    def __init__(self) -> None:
        super().__init__('napari')
        self.injection_store.namespace = _napari_names
        _init_processors(store=self.injection_store)
        _init_providers(store=self.injection_store)

        for action in LAYER_ACTIONS:
            self.register_action(action)

        self.menus.append_menu_items(SUBMENUS)

    def inject_dependencies(
        self,
        func: Callable[..., C],
        *,
        localns: Optional[dict] = None,
        on_unresolved_required_args: RaiseWarnReturnIgnore = "raise",
        on_unannotated_required_args: RaiseWarnReturnIgnore = "warn",
    ) -> Callable[..., C]:
        """Decorator returns func that can access/process napari objects based on type hints.

        This is form of dependency injection, and result processing.  It does 2 things:

        1. If `func` includes a parameter that has a type with a registered provider
        (e.g. `Viewer`, or `Layer`), then this decorator will return a new version of
        the input function that can be called *without* that particular parameter.

        2. If `func` has a return type with a registered processor (e.g. `ImageData`),
        then this decorator will return a new version of the input function that, when
        called, will have the result automatically processed by the current processor
        for that type (e.g. in the case of `ImageData`, it will be added to the viewer.)

        Parameters
        ----------
        func : Callable
            A function with napari type hints.

        Returns
        -------
        Callable
            A function with napari dependencies injected
        """
        return ino.inject_dependencies(
            func,
            localns=localns,
            store=self.injection.store,
            on_unresolved_required_args=on_unresolved_required_args,
            on_unannotated_required_args=on_unannotated_required_args,
        )


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


app = NapariApplication()
