from __future__ import annotations

from functools import lru_cache
from typing import Dict

from app_model import Application

from ._submenus import SUBMENUS
from .actions._layer_actions import LAYER_ACTIONS
from .injection._processors import PROCESSORS
from .injection._providers import PROVIDERS


class NapariApplication(Application):
    def __init__(self) -> None:
        super().__init__('napari')
        self.injection_store.namespace = _napari_names  # type: ignore [assignment]
        self.injection_store.register(
            providers=PROVIDERS, processors=PROCESSORS
        )

        for action in LAYER_ACTIONS:
            self.register_action(action)

        self.menus.append_menu_items(SUBMENUS)


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
