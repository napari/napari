import os

import numpy as np

from napari.layers import Tracks

from .utils import Skiper


class TracksSuite:
    param_names = ['size', 'n_tracks']
    params = [(5 * np.power(10, np.arange(7))).tolist(), [1, 10, 100, 1000]]

    if "PR" in os.environ:
        skip_params = Skiper(lambda x: x[0] > 500 or x[1] > 10)

    def setup(self, size, n_tracks):
        """
        Create tracks data
        """

        if 5 * n_tracks > size:
            # not useful, tracks to short or larger than size
            raise NotImplementedError

        rng = np.random.default_rng(0)

        track_ids = rng.integers(n_tracks, size=size)
        time = np.zeros(len(track_ids))

        for value, counts in zip(*np.unique(track_ids, return_counts=True)):
            t = rng.permutation(counts)
            time[track_ids == value] = t

        coordinates = rng.uniform(size=(size, 3))

        data = np.concatenate(
            (track_ids[:, None], time[:, None], coordinates),
            axis=1,
        )

        self.data = data

    def time_create_layer(self, *args) -> None:
        Tracks(self.data)
