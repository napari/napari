import pytest

from napari._qt.qt_application import NapariQApplication


@pytest.fixture(scope="session")
def qapp():
    yield NapariQApplication([])
