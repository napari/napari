"""Write files in the chrome://tracing format.
"""
import json
import os
from pathlib import Path
import threading
from typing import Union


class ChromeTracingFile:
    """Writes a Chrome tracing file.

    There are two chrome://tracing styles:
    1) JSON Array Format
    2) JSON Object Format

    We are using style 1 for now since you can stop/truncate the file at
    anytime.

    Both formats are essentially JSON, see the "trace_event format" Google Doc
    linked from this page:
    https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md

    Parameters
    ----------
    path : str
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.

    """

    def __init__(self, path: Union[Path, str]):
        """Open the tracing file on disk."""
        # PID goes in every event.
        self.pid = os.getpid()

        # TID goes in every event, for now assume the current thread.
        self.tid = threading.get_ident()

        # Catagories can be toggled on/off in the UI. For now we only have
        # one category, they are all Qt Events.
        self.cat = "qt_event"

        # Start the JSON Array Format with an open bracket.
        self.outf = open(path, "w")
        self.outf.write("[\n")

    def write_event(self, name, start_us, duration_us):
        """Write one completed event."""
        data = {
            "pid": self.pid,
            "name": name,
            "cat": self.cat,
            "ph": "X",  # X is a "completed event"
            "ts": start_us,
            "dur": duration_us,
        }

        # Write comma separated JSON objects.
        self.outf.write(f"{json.dumps(data)},\n")
