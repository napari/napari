"""Write files in the chrome://tracing file format.
"""
import json
import os
import threading


class ChromeTracingFile:
    """Writes a Chrome tracing file.

    There are two chrome://tracing styles:
    1) JSON Array Format
    2) JSON Object Format

    We are using style 1 for now since you can stop/truncate the file and its
    still valid. See the "trace_event format" Google Doc on this page:
    https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md
    """

    def __init__(self, path):
        """Open the tracing file on disk."""
        self.pid = os.getpid()
        self.tid = threading.get_ident()

        # Catagories can be toggle on/off in the UI, for now all our events
        # are Qt events, but we will have more types later.
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
