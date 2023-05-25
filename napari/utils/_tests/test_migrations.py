import pytest

from napari.utils.migrations import rename_argument


def test_simple():
    @rename_argument("a", "b", "1")
    def sample_fun(b):
        return b

    assert sample_fun(1) == 1
    assert sample_fun(b=1) == 1
    with pytest.deprecated_call():
        assert sample_fun(a=1) == 1
    with pytest.raises(ValueError):
        sample_fun(b=1, a=1)


def test_constructor():
    class Sample:
        @rename_argument("a", "b", "1")
        def __init__(self, b) -> None:
            self.b = b

    assert Sample(1).b == 1
    assert Sample(b=1).b == 1
    with pytest.deprecated_call():
        assert Sample(a=1).b == 1
