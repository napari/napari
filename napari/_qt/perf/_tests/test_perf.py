import json
import os
import subprocess
import sys
from pathlib import Path

from napari._tests.utils import skip_local_popups

# NOTE:
# for some reason, running this test fails in a subprocess with a segfault
# if you don't show the viewer...
PERFMON_SCRIPT = """
import napari
from qtpy.QtCore import QTimer

v = napari.view_points()
QTimer.singleShot(100, napari._qt.qt_event_loop.quit_app)
napari.run()
"""

CONFIG = {
    "trace_qt_events": True,
    "trace_file_on_start": '',
    "trace_callables": ["chunk_loader"],
    "callable_lists": {
        "chunk_loader": [
            "napari.components.experimental.chunk._loader.ChunkLoader.load_request",
            "napari.components.experimental.chunk._loader.ChunkLoader._on_done",
        ]
    },
}


@skip_local_popups
def test_trace_on_start(tmp_path: Path):
    """Make sure napari can write a perfmon trace file."""
    trace_path = tmp_path / "trace.json"
    config_path = tmp_path / "perfmon.json"
    CONFIG['trace_file_on_start'] = str(trace_path)
    config_path.write_text(json.dumps(CONFIG))

    env = os.environ.copy()
    env.update({'NAPARI_PERFMON': str(config_path), 'NAPARI_CONFIG': ''})
    subprocess.run([sys.executable, '-c', PERFMON_SCRIPT], env=env, check=True)

    # Make sure file exists and is not empty.
    assert trace_path.exists(), "Trace file not written"
    assert trace_path.stat().st_size > 0, "Trace file is empty"

    # Assert every event contains every important field.
    with open(trace_path) as infile:
        data = json.load(infile)
        assert len(data) > 0
        for event in data:
            for field in ['pid', 'tid', 'name', 'ph', 'ts', 'args']:
                assert field in event
