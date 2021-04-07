from typing import Optional

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """An object to store the provenance of a layer."""

    # filpath/url associated with layer
    path: Optional[str] = None
    # name of plugin that loaded the file (if applic.)
    plugin: Optional[str] = None
    # fully qualified name of the highest level API method that
    # created the layer (e.g. viewer.open)
    method: Optional[str] = None
