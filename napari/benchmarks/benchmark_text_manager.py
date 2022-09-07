# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://napari.org/developers/benchmarks.html
import numpy as np
import pandas as pd

from napari.layers.utils.text_manager import TextManager


class TextManagerSuite:
    """Benchmarks for creating and modifying a text manager."""

    param_names = ['n', 'string']
    params = [
        [2**i for i in range(4, 18, 2)],
        [
            {'constant': 'test'},
            'string_property',
            'float_property',
            '{string_property}: {float_property:.2f}',
        ],
    ]

    def setup(self, n, string):
        np.random.seed(0)
        categories = ('cat', 'car')
        self.features = pd.DataFrame(
            {
                'string_property': pd.Series(
                    np.random.choice(categories, n),
                    dtype=pd.CategoricalDtype(categories),
                ),
                'float_property': np.random.rand(n),
            }
        )
        self.current_properties = self.features.iloc[[-1]].to_dict('list')
        self.manager = TextManager(string=string, features=self.features)
        self.indices_to_remove = list(range(0, n, 2))

    def time_create(self, n, string):
        TextManager(string=string, features=self.features)

    def time_refresh(self, n, string):
        self.manager.refresh_text(self.features)

    def time_add_iteratively(self, n, string):
        for _ in range(512):
            self.manager.add(self.current_properties, 1)

    def time_remove_as_batch(self, n, string):
        self.manager.remove(self.indices_to_remove)

    # `time_remove_as_batch` can only run once per instance;
    # otherwise it fails because the indices were already removed:
    #
    #   IndexError: index 32768 is out of bounds for axis 0 with size 32768
    #
    # Why? ASV will run the same function after setup several times in two
    # occasions: warmup and timing itself. We disable warmup and only
    # allow one execution per state with these method-specific options:
    time_remove_as_batch.number = 1
    time_remove_as_batch.warmup_time = 0
    # See https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks
    # for more details
