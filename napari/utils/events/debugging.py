import inspect
import os
import site
from textwrap import indent
from typing import TYPE_CHECKING, ClassVar, Set

from pydantic import BaseSettings, Field, PrivateAttr

from ...utils.misc import ROOT_DIR

try:
    from rich import print
except ImportError:
    print("TIP: run `pip install rich` for much nicer event debug printout.")


if TYPE_CHECKING:
    from .event import Event


class EventDebugSettings(BaseSettings):
    include_emitters: Set[str] = Field(default_factory=set)
    include_events: Set[str] = Field(default_factory=set)
    exclude_emitters: Set[str] = {'TransformChain', 'Context'}
    exclude_events: Set[str] = {'status', 'position'}
    stack_depth: int = 6

    _cur_depth: ClassVar[int] = PrivateAttr(0)

    class Config:
        env_file = '.env'
        env_prefix = 'event_debug_'


_SETTINGS = EventDebugSettings()
_SP = site.getsitepackages()[0]
_STD_LIB = site.__file__.rsplit(os.path.sep, 1)[0]


def _shorten_fname(fname: str) -> str:
    """Reduce extraneous stuff from filenames"""
    fname = fname.replace(_SP, '.../site-packages')
    fname = fname.replace(_STD_LIB, '.../python')
    return fname.replace(ROOT_DIR, "napari")


def log_event_stack(event: 'Event', cfg: EventDebugSettings = _SETTINGS):
    """Print info about what caused this event to be emitted.s"""

    if cfg.include_events:
        if event.type not in cfg.include_events:
            return
    elif event.type in cfg.exclude_events:
        return

    source = type(event.source).__name__
    if cfg.include_emitters:
        if source not in cfg.include_emitters:
            return
    elif source in cfg.exclude_emitters:
        return

    # get values being emitted
    vals = ",".join(f"{k}={v}" for k, v in event._kwargs.items())
    # show event type and source
    lines = [f'{source}.events.{event.type}({vals})', '  was triggered by:']
    # climb stack and show what caused it.
    # note, we start 2 frames back in the stack, one frame for *this* function
    # and the second frame for the EventEmitter.__call__ function (where this
    # function was likely called).
    for frame in inspect.stack(0)[2 : 2 + cfg.stack_depth]:
        fname = _shorten_fname(frame.filename)
        ln = f'  "{fname}", line {frame.lineno}, in {frame.function}'
        lines.append(ln)
    lines.append("")

    # seperate groups of events
    if not cfg._cur_depth:
        lines = ["─" * 79, ''] + lines

    # log it
    print(indent("\n".join(lines), '  ' * cfg._cur_depth))

    # spy on nested events...
    # (i.e. events that were emitted while another was being emitted)
    def _pop_source():
        cfg._cur_depth -= 1
        return event._sources.pop()

    event._pop_source = _pop_source
    cfg._cur_depth += 1
