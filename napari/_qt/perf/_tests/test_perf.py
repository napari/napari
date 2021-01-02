import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from napari._qt.perf.qt_event_tracing import QApplicationWithTracing
from napari._qt.utils import delete_qapp
from napari.utils.perf import get_perf_config, timers


@pytest.fixture(scope="module")
def qapp():
    if QApplication.instance() is not None:
        delete_qapp(QApplication.instance())
    yield QApplicationWithTracing([])


@contextmanager
def clear_perf_context():
    """Yield a fresh perf config."""
    get_perf_config.cache_clear()
    yield
    get_perf_config.cache_clear()


def test_perfmon_off_by_default():
    """Make sure perfmon off by default."""
    # Check perfmon is not enabled
    with clear_perf_context():
        assert get_perf_config() is None


def test_trace_on_start(tmp_path, monkeypatch, make_test_viewer):
    """Make sure napari can write a perfmon trace file."""

    monkeypatch.setenv('NAPARI_PERFMON', '1')
    # Check perfmon is enabled
    with clear_perf_context():
        assert get_perf_config()

        trace_path = tmp_path / "trace.json"
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
