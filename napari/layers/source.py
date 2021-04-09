from dataclasses import field
from typing import Optional

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """An object to store the provenance of a layer.

    Parameters
    ----------
    path: str, optional
        filpath/url associated with layer
    plugin: str, optional
        name of plugin that loaded the file (if applicable)
    method: str, optional
        fully qualified name of the highest level API method that created the
        layer (e.g. viewer.open)
    """

    path: Optional[str] = None
    plugin: Optional[str] = None
    method: Optional[str] = None
    kwargs: Optional[dict] = field(default_factory=dict)
