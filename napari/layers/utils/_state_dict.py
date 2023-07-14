import warnings
from typing import Any


class LayerStateDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        # Maps from deprecations keys to their deprecation messages.
        self.deprecations: dict[str, str] = {}
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: Any) -> Any:
        if message := self.deprecations.get(key):
            warnings.warn(message, DeprecationWarning)
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if message := self.deprecations.get(key):
            warnings.warn(message, DeprecationWarning)
        return super().__setitem__(key, value)
