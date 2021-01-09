from pathlib import Path
import napari
import pytest
import runpy


# not testing these examples
skip = [
    'surface_timeseries.py',  # needs nilearn
    '3d_kymograph.py',  # needs tqdm
    'live_tiffs.py',  # requires files
    'live_tiffs_generator.py',
]
example_dir = Path(napari.__file__).parent.parent / 'examples'
examples = [f for f in example_dir.glob("*.py") if f.name not in skip]


@pytest.fixture
def qapp():
    from napari._qt.qt_event_loop import get_app
    from qtpy.QtCore import QTimer

    # it's important that we use get_app so that it connects to the
    # app.aboutToQuit.connect(wait_for_workers_to_quit)
    app = get_app()

    # quit examples that explicitly start the event loop with `napari.run()`
    # so that tests aren't waiting on a manual exit
    QTimer.singleShot(100, app.quit)

    yield app


@pytest.mark.parametrize("fname", examples, ids=lambda x: Path(x).name)
def test_examples(qapp, fname, monkeypatch, capsys):
    """Test that all of our examples are still working without warnings."""

    from napari._qt.qt_main_window import Window
    from napari._qt.exceptions import ExceptionHandler

    # hide viewer window
    monkeypatch.setattr(Window, 'show', lambda *a: None)

    # make sure our sys.excepthook override in gui_qt doesn't hide errors
    def raise_errors(self, etype, value, tb):
        raise value

    monkeypatch.setattr(ExceptionHandler, 'handle', raise_errors)

    # run the example!
    assert runpy.run_path(str(fname))
