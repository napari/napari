import numpy as np

from napari.layers import Tracks

from .utils import Skipper


class TracksSuite:
    param_names = ['size', 'n_tracks']
    params = [(5 * np.power(10, np.arange(7))).tolist(), [1, 10, 100, 1000]]

    skip_params = Skipper(
        func_pr=lambda x: x[0] > 500 or x[1] > 10,
        func_always=lambda x: x[1] * 5 > x[0],
    )
    # we skip cases where the number of tracks times five is larger than the size as it is not useful

    def setup(self, size, n_tracks):
        """
        Create tracks data
        """
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

    def time_create_layer(self, *_) -> None:
        Tracks(self.data)
