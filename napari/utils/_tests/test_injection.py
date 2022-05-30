import pytest

from napari.components import LayerList
from napari.utils._injection import inject_napari_dependencies, set_accessor


def test_napari_injection():
    @inject_napari_dependencies
    def f(ll: LayerList):
        return ll

    some_layers = LayerList()

    assert f() is None

    with set_accessor({LayerList: lambda: some_layers}, clobber=True):
        assert f() is some_layers

    assert f() is None


def test_napari_injection_missing():
    @inject_napari_dependencies
    def f(x: int):
        return x

    assert f(4) == 4

    with pytest.raises(TypeError):
        f()

    with set_accessor({int: lambda: 1}):
        assert f() == 1
