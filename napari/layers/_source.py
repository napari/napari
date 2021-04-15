from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Tuple

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """An object to store the provenance of a layer.

    Parameters
    ----------
    path: str, optional
        filpath/url associated with layer
    reader_plugin: str, optional
        name of reader plugin that loaded the file (if applicable)
    sample: Tuple[str, str], optional
        Tuple of (sample_plugin, sample_name), if layer was loaded via the
        open_sample.
    """

    path: Optional[str] = None
    reader_plugin: Optional[str] = None
    sample: Optional[Tuple[str, str]] = None


# layer source context management

_LAYER_SOURCE: ContextVar[dict] = ContextVar('_LAYER_SOURCE', default={})


@contextmanager
def layer_source(**source_kwargs):
    """Creates context in which all layers will be given `source_kwargs`.

    Parameters
    ----------
    **source_kwargs :
        keys/values should be valid parameters for :class:`Source`.
    """
    prev = _LAYER_SOURCE.get()
    _LAYER_SOURCE.set({**prev, **source_kwargs})
    try:
        yield
    finally:
        _LAYER_SOURCE.set(prev)


def current_source():
    """Get the current :class:`Source` (inferred from context)."""
    return Source(**_LAYER_SOURCE.get())
