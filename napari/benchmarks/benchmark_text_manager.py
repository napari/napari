# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://napari.org/developers/benchmarks.html
import numpy as np

from napari.layers.utils.text_manager import TextManager


class TextManagerSuite:
    """Benchmarks for creating and modifying a text manager."""

    param_names = ['n', 'text']
    params = [
        [2 ** i for i in range(4, 18, 2)],
        [
            None,
            'constant',
            'string_property',
            'float_property',
            '{string_property}: {float_property:.2f}',
        ],
    ]

    def setup(self, n, text):
        np.random.seed(0)
        self.properties = {
            'string_property': np.random.choice(('cat', 'car'), n),
            'float_property': np.random.rand(n),
        }
        self.current_properties = {
            k: np.array([v[-1]]) for k, v in self.properties.items()
        }
        self.manager = TextManager(
            n_text=n, properties=self.properties, text=text
        )
        self.indices_to_remove = list(range(0, n, 2))

    def time_create(self, n, text):
        TextManager(n_text=n, properties=self.properties, text=text)

    def time_refresh(self, n, text):
        self.manager.refresh_text(self.properties)

    def time_add_iteratively(self, n, text):
        for _ in range(512):
            self.manager.add(self.current_properties, 1)

    def time_remove_as_batch(self, n, text):
        self.manager.remove(self.indices_to_remove)
