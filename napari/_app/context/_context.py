from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Optional

from app_model.expressions import Context
from app_model.expressions import create_context as _create_context
from app_model.expressions import get_context

from ...utils.translations import trans

if TYPE_CHECKING:
    from ...utils.events import Event

__all__ = ["create_context", "get_context", "Context", "SettingsAwareContext"]


class SettingsAwareContext(Context):
    """A special context that allows access of settings using `settings.`

    This takes no parents, and will always be a root context.
    """

    _PREFIX: Final[str] = 'settings.'

    def __init__(self):
        super().__init__()
        from ...settings import get_settings

        self._settings = get_settings()
        self._settings.events.changed.connect(self._update_key)

    def _update_key(self, event: Event):
        self.changed.emit({f'{self._PREFIX}{event.key}'})

    def __del__(self):
        self._settings.events.changed.disconnect(self._update_key)

    def __missing__(self, key: str) -> Any:
        if key.startswith(self._PREFIX):
            splits = [k for k in key.split(".")[1:] if k]
            val: Any = self._settings
            if splits:
                while splits:
                    val = getattr(val, splits.pop(0))
                if hasattr(val, 'dict'):
                    val = val.dict()
                return val
        return super().__missing__(key)

    def new_child(self, m: Optional[dict] = None) -> Context:  # type: ignore
        """New ChainMap with a new map followed by all previous maps.

        If no map is provided, an empty dict is used.
        """
        # important to use self, not *self.maps
        return Context(m or {}, self)  # type: ignore

    def __setitem__(self, k: str, v: Any) -> None:
        if k.startswith(self._PREFIX):
            raise ValueError(
                trans._(
                    "Cannot set key starting with {prefix!r}",
                    deferred=True,
                    prefix=self._PREFIX,
                )
            )

        return super().__setitem__(k, v)

    def __bool__(self):
        return True


def create_context(
    obj: object,
    max_depth: int = 20,
    start: int = 2,
    root: Optional[Context] = None,
) -> Optional[Context]:
    return _create_context(
        obj=obj,
        max_depth=max_depth,
        start=start,
        root=root,
        root_class=SettingsAwareContext,
    )
