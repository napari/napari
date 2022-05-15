from napari.components import LayerList
from napari.utils._injection import inject_napari_dependencies, set_accessor


def test_napari_injection():
    @inject_napari_dependencies
    def f(ll: LayerList):
        return ll

    some_layers = LayerList()
    with set_accessor({LayerList: lambda: some_layers}):
        assert f() is some_layers
