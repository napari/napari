from contextlib import contextmanager

import numpy as np

from napari.utils import progrange, progress


@contextmanager
def assert_progress_added_to_all(prog):
    """Check prog is added to `progress_instances` on init & removed on close"""
    assert prog in progress._all_instances
    yield
    assert prog not in progress._all_instances


def test_progress_with_iterable():
    """Test typical iterable is correctly built"""
    r = range(100)
    pbr = progress(r, desc='iterable')
    assert pbr.iterable is r
    assert pbr.n == 0

    with assert_progress_added_to_all(pbr):
        pbr.close()


def test_progress_with_ndarray():
    """Test 2D ndarray is correctly built"""
    iter_ = np.random.random((100, 100))
    pbr = progress(iter_, desc='ndarray')

    assert pbr.iterable is iter_
    assert pbr.n == 0

    with assert_progress_added_to_all(pbr):
        pbr.close()


def test_progress_with_total():
    """Test progress with total not iterable, and manual updates"""
    pbr = progress(total=5, desc='total')

    assert pbr.n == 0
    pbr.update(1)
    assert pbr.n == 1

    with assert_progress_added_to_all(pbr):
        pbr.close()


def test_progress_with_context():
    """Test context manager works as expected"""
    with progress(range(100), desc='context') as pbr:
        assert pbr in progress._all_instances
        assert pbr.n == 0
    assert pbr not in progress._all_instances


def test_progress_update():
    """Test update with different values"""
    pbr = progress(total=10, desc='update')
    assert pbr.n == 0

    pbr.update(1)
    pbr.refresh()  # not sure why this has to be called manually here
    assert pbr.n == 1

    pbr.update(2)
    pbr.refresh()
    assert pbr.n == 3

    pbr.update()
    pbr.refresh()
    assert pbr.n == 4

    with assert_progress_added_to_all(pbr):
        pbr.close()


def test_progress_set_description():
    """Test setting description works as expected"""
    pbr = progress(total=5)
    pbr.set_description("Test")

    assert pbr.desc == "Test: "

    pbr.close()
    assert pbr not in progress._all_instances


def test_progrange():
    """Test progrange shorthand for progress(range(n))"""
    with progrange(10) as pbr, progress(range(10)) as pbr2:
        assert pbr.iterable == pbr2.iterable
    assert pbr not in progress._all_instances
