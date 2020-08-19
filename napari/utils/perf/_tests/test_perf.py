import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


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


def _create_config(trace_path) -> dict:
    """Return config file that traces to teh given path."""
    return {
        "trace_qt_events": True,
        "trace_file_on_start": trace_path,
    }


def _write_config_file(config_file, trace_file) -> None:
    """Write a config file that traces to the given trace_file."""
    with open(config_file, 'w') as outf:
        json.dump(_create_config(trace_file), outf)


def _trace_file_okay(trace_path: str) -> bool:
    """For now okay just means valid JSON and not empty."""
    with open(trace_path) as infile:
        print(infile.readlines())
        data = json.load(infile)
        return data.keys() > 1


def test_trace_on_start(make_test_viewer):
    """Make sure napari creates a trace file when perfmon is enabled."""
    with temporary_file('json') as config_path:
        os.environ['NAPARI_PERFMON'] = config_path
        with temporary_file('json') as trace_path:
            _write_config_file(config_path, trace_path)
            viewer = make_test_viewer()
            data = np.random.random((10, 15))
            viewer.add_image(data)
            viewer.close()

            assert Path(trace_path).stat().st_size > 0, "Trace file is empty"
            assert _trace_file_okay(trace_path)
