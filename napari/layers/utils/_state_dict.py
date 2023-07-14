import warnings
from typing import Any


class LayerStateDict(dict[str, Any]):
    def __init__(self, *args, **kwargs) -> None:
        # Maps from deprecations keys to their deprecation messages.
        self.deprecations: dict[str, str] = {}
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        if message := self.deprecations.get(key):
            warnings.warn(message, DeprecationWarning)
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if message := self.deprecations.get(key):
            warnings.warn(message, DeprecationWarning)
        return super().__setitem__(key, value)
