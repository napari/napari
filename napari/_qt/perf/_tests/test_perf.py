import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import qtpy
from qtpy.QtWidgets import QApplication

from napari._qt.perf import qt_event_tracing
from napari._qt.utils import delete_qapp
from napari.utils import perf

if os.getenv('CI', '0') != '0' and (
    sys.version_info >= (3, 9)
    or (
        (sys.platform.startswith('linux') or sys.platform.startswith('win'))
        and qtpy.API_NAME == "PySide2"
    )
):
    # this test is covered by other platforms, and also seems to work locally
    # on linux
    pytest.skip(
        "Perfmon segfaults on linux or windows CI with pyside2, or with python 3.9",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def qapp():
    """A modified QApplicationWithTracing just for this test module.

    Note: Because of the difficulty in destroying a QApplication that overrides
    .notify() like QApplicationWithTracing does, this test must be run last
    (globally).  So in napari/conftest.py::pytest_collection_modifyitems,
    we ensure that this test is always last.
    """

    # before creating QApplicationWithTracing, we need to monkeypatch
    # the `perf.perf_timer` context manager that gets used in the
    # qt_event_tracing module

    original_perf_timer = qt_event_tracing.perf.perf_timer
    _, perf_timer, _, _ = perf._timers._create_timer()
    qt_event_tracing.perf.perf_timer = perf_timer

    try:
        if qtpy.API_NAME == 'PySide2' and QApplication.instance():
            delete_qapp(QApplication.instance())
        yield qt_event_tracing.QApplicationWithTracing([])
    finally:
        qt_event_tracing.perf.perf_timer = original_perf_timer


def test_trace_on_start(tmp_path, monkeypatch, make_napari_viewer):
    """Make sure napari can write a perfmon trace file."""

    timers, _, _, _ = perf._timers._create_timer()
    monkeypatch.setattr(perf._timers, 'timers', timers)

    # Check perfmon is enabled
    trace_path = tmp_path / "trace.json"
    timers.start_trace_file(trace_path)

    viewer = make_napari_viewer()
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
