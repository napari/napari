import pytest

from napari.utils.migrations import add_deprecated_property, rename_argument


def test_simple():
    @rename_argument("a", "b", "1", "0.5")
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
        @rename_argument("a", "b", "1", "0.5")
        def __init__(self, b) -> None:
            self.b = b

    assert Sample(1).b == 1
    assert Sample(b=1).b == 1
    with pytest.deprecated_call():
        assert Sample(a=1).b == 1


def test_deprecated_property() -> None:
    class Dummy:
        def __init__(self) -> None:
            self._value = 0

        @property
        def new_property(self) -> int:
            return self._value

        @new_property.setter
        def new_property(self, value: int) -> int:
            self._value = value

    instance = Dummy()

    add_deprecated_property(
        Dummy, "old_property", "new_property", "0.1.0", "0.0.0"
    )

    assert instance.new_property == 0

    instance.new_property = 1

    msg = "Dummy.old_property is deprecated since 0.0.0 and will be removed in 0.1.0. Please use new_property"

    with pytest.warns(FutureWarning, match=msg):
        assert instance.old_property == 1

    with pytest.warns(FutureWarning, match=msg):
        instance.old_property = 2

    assert instance.new_property == 2
