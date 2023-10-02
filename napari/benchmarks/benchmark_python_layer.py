from napari.layers.points._points_utils import coerce_symbols


class CoerceSymbolsSuite:
    def setup(self):
        self.symbols1 = ['o' for _ in range(10**6)]
        self.symbols2 = ['o' for _ in range(10**6)]
        self.symbols2[10000] = 's'

    def time_coerce_symbols1(self):
        coerce_symbols(self.symbols1)

    def time_coerce_symbols2(self):
        coerce_symbols(self.symbols2)
