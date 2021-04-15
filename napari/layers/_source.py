from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional, Tuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from magicgui.widgets import FunctionGui


class Source(BaseModel):
    """An object to store the provenance of a layer.

    Parameters
    ----------
    path: str, optional
        filpath/url associated with layer
    reader_plugin: str, optional
        name of reader plugin that loaded the file (if applicable)
    sample: Tuple[str, str], optional
        Tuple of (sample_plugin, sample_name), if layer was loaded via
        `viewer.open_sample`.
    widget: FunctionGui, optional
        magicgui widget, if the layer was added via a magicgui widget.
    """

    path: Optional[str] = None
    reader_plugin: Optional[str] = None
    sample: Optional[Tuple[str, str]] = None
    widget: Optional[FunctionGui] = None

    class Config:
        arbitrary_types_allowed = True
        frozen = True


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
    token = _LAYER_SOURCE.set({**_LAYER_SOURCE.get(), **source_kwargs})
    try:
        yield
    finally:
        _LAYER_SOURCE.reset(token)


def current_source():
    """Get the current layer :class:`Source` (inferred from context)."""
    return Source(**_LAYER_SOURCE.get())
