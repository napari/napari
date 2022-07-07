import sys
import os

import pytest

# check if this module has been explicitly requested or `--test-examples` is included
fpath = os.path.join(*__file__.split(os.path.sep)[-3:])
if '--test-examples' not in sys.argv and fpath not in sys.argv:
    pytest.skip(
        'Use `--test-examples` to test examples', allow_module_level=True
    )

import runpy
from pathlib import Path

import numpy as np
import skimage.data
from qtpy import API_NAME

import napari
from napari._qt.qt_main_window import Window
from napari.utils.notifications import notification_manager

# not testing these examples
skip = [
    'surface_timeseries_.py',  # needs nilearn
    '3d_kymograph_.py',  # needs tqdm
    'live_tiffs_.py',  # requires files
    'tiled-rendering-2d_.py',  # too slow
    'live_tiffs_generator_.py',
    'points-over-time.py',  # too resource hungry
    'embed_ipython_.py',  # fails without monkeypatch
    'new_theme.py',  # testing theme is extremely slow on CI
    'dynamic-projections-dask.py',  # extremely slow / does not finish
]

EXAMPLE_DIR = Path(napari.__file__).parent.parent / 'examples'
# using f.name here and re-joining at `run_path()` for test key presentation
# (works even if the examples list is empty, as opposed to using an ids lambda)
examples = [f.name for f in EXAMPLE_DIR.glob("*.py") if f.name not in skip]

# still some CI segfaults, but only on windows with pyqt5
if os.getenv("CI") and os.name == 'nt' and API_NAME == 'PyQt5':
    examples = []

if os.getenv("CI") and os.name == 'nt' and 'to_screenshot.py' in examples:
    examples.remove('to_screenshot.py')


@pytest.mark.filterwarnings("ignore")
@pytest.mark.skipif(not examples, reason="No examples were found.")
@pytest.mark.parametrize("fname", examples)
def test_examples(builtins, fname, monkeypatch):
    """Test that all of our examples are still working without warnings."""

    # hide viewer window
    monkeypatch.setattr(Window, 'show', lambda *a: None)
    # prevent running the event loop
    monkeypatch.setattr(napari, 'run', lambda *a, **k: None)
    # Prevent downloading example data because this sometimes fails.
    monkeypatch.setattr(
        skimage.data,
        'cells3d',
        lambda: np.zeros((60, 2, 256, 256), dtype=np.uint16),
    )

    # make sure our sys.excepthook override doesn't hide errors
    def raise_errors(etype, value, tb):
        raise value

    monkeypatch.setattr(notification_manager, 'receive_error', raise_errors)

    # run the example!
    try:
        runpy.run_path(str(EXAMPLE_DIR / fname))
    except SystemExit as e:
        # we use sys.exit(0) to gracefully exit from examples
        if e.code != 0:
            raise
    finally:
        napari.Viewer.close_all()
