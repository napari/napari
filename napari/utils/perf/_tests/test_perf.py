import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from .. import timers


@contextmanager
def temporary_file(suffix=''):
    """Yield a writeable temporary filename that is deleted on context exit.
    Parameters
    ----------
    suffix : string, optional
        The suffix for the file.
    """
    tempfile_stream = NamedTemporaryFile(suffix=suffix, delete=False)
    tempfile = tempfile_stream.name
    tempfile_stream.close()
    yield tempfile
    os.remove(tempfile)


def _trace_file_okay(trace_path: str) -> bool:
    """For now okay just means valid JSON and not empty."""


@pytest.mark.perfmon
def test_trace_on_start(make_test_viewer):
    """Make sure napari can write a perfmon trace file."""
    with temporary_file('json') as trace_path:
        timers.start_trace_file(trace_path)

        viewer = make_test_viewer()
        data = np.random.random((10, 15))
        viewer.add_image(data)
        viewer.close()

        timers.stop_trace_file()

        # Make sure file exists and is not empty.
        assert Path(trace_path).exists(), "Trace file not written"
        assert Path(trace_path).stat().st_size > 0, "Trace file is empty"

        # Assert every event contains every important field.
        with open(trace_path) as infile:
            data = json.load(infile)
            assert len(data) > 0
            for event in data:
                for field in ['pid', 'tid', 'name', 'ph', 'ts', 'args']:
                    assert field in event
