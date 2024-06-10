import warnings
from typing import Any

from napari.utils.translations import trans


class DeprecatingDict(dict[str, Any]):
    # Maps from deprecations keys to their deprecation messages.
    deprecations: dict[str, str]

    def __init__(self, *args, **kwargs) -> None:
        self.deprecations = {}
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        if message := self.deprecations.get(key):
            warnings.warn(message, DeprecationWarning)
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if message := self.deprecations.get(key):
            warnings.warn(message, DeprecationWarning)
        return super().__setitem__(key, value)

    def deprecate(
        self,
        key: str,
        *,
        new_key: str,
        version: str,
        since_version: str,
    ) -> None:
        """Deprecates a key with a new key using an associated default message."""
        message = trans._(
            '{key} is deprecated since {since_version} and will be removed in {version}. Please use {new_key}',
            deferred=True,
            key=key,
            since_version=since_version,
            version=version,
            new_key=new_key,
        )
        self.deprecations[key] = message
