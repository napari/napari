import numpy as np

from napari.layers.tracks._managers import (
    InteractiveTrackManager,
    TrackManager,
)


class _BaseTrackManagerSuite:
    param_names = ['size', 'n_tracks']
    params = [(5 * np.power(10, np.arange(2, 7))).tolist(), [10, 100, 1000]]

    def setup(self, size, n_tracks):
        """
        Create tracks data
        """
        if 10 * n_tracks > size:
            # number of tracks are adjusted if not 10 times larger than size
            n_tracks = size // 10

        rng = np.random.default_rng()

        track_ids = rng.integers(1, n_tracks + 1, size=size)
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


class TrackManagerSuite(_BaseTrackManagerSuite):
    def time_default_manager(self, *args):
        manager = TrackManager()
        manager.data = self.data
        _ = manager.data

    def time_interactive_manager(self, *args):
        """Time to initialize and serialize data."""
        manager = InteractiveTrackManager(data=self.data)
        _ = manager.data


class InteractiveTrackManagerSuite(_BaseTrackManagerSuite):
    param_names = ['size', 'time_window']
    params = [
        (5 * np.power(10, np.arange(2, 7))).tolist(),
        [10, 50, 100, 200, 400],
    ]

    def setup(self, size, time_window):
        """Time to initialize the data."""
        super().setup(size=size, n_tracks=100)
        self.manager = InteractiveTrackManager(data=self.data)

    def time_interactive_serialization(self, size, time_window):
        """Time to serialize data without initialization."""
        self.manager._serialize(0, 0 + time_window)
