import numpy as np

from napari.utils import progrange, progress


def test_progress_with_iterable():
    r = range(100)
    pbr = progress(r, desc='iterable')
    assert pbr.iterable is r
    assert pbr.n == 0

    pbr.close()
    assert pbr not in progress.all_progress


def test_progress_with_ndarray():
    iter_ = np.random.random((100, 100))
    pbr = progress(iter_, desc='ndarray')

    assert pbr.iterable is iter_
    assert pbr.n == 0

    pbr.close()
    assert pbr not in progress.all_progress


def test_progress_with_total():
    pbr = progress(total=5, desc='total')

    assert pbr.n == 0
    pbr.update(1)
    assert pbr.n == 1

    pbr.close()
    assert pbr not in progress.all_progress


def test_progress_with_context():
    with progress(range(100), desc='context') as pbr:
        assert pbr.n == 0
    assert pbr not in progress.all_progress


def test_progress_update():
    pbr = progress(total=10, desc='update')
    assert pbr.n == 0

    pbr.update(1)
    pbr.refresh()  # not sure why this has to be called manually here
    assert pbr.n == 1

    pbr.update(2)
    pbr.refresh()
    assert pbr.n == 3

    pbr.close()
    assert pbr not in progress.all_progress


def test_progress_set_description():
    pbr = progress(total=5)
    pbr.set_description("Test")

    assert pbr.desc == "Test: "

    pbr.close()
    assert pbr not in progress.all_progress


def test_progrange():
    with progrange(10) as pbr:
        with progress(range(10)) as pbr2:
            assert pbr.iterable == pbr2.iterable
    assert pbr not in progress.all_progress
