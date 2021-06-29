import pytest

from .._plane import Plane


@pytest.fixture
def plane():
    return Plane(position=(64, 64, 64), normal=(1, 0, 0))
