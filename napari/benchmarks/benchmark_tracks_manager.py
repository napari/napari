import numpy as np

from napari.layers.tracks._managers import (
    InteractiveTrackManager,
    TrackManager,
)


class _BaseTrackManagerSuite:
    param_names = ['size', 'sorted', 'n_tracks']
    params = [np.power(10, np.arange(2, 7)).tolist(), [False, True], [10, 100]]

    def setup(self, size, sorted, n_tracks):
        """
        Create tracks data
        """
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

        if sorted:
            order = np.lexsort((time, track_ids))
            data = data[order]

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
    def setup(self, size, sorted, n_tracks):
        super().setup(size=size, sorted=sorted, n_tracks=n_tracks)
        self.manager = InteractiveTrackManager(data=self.data)

    def time_interactive_serialization(self, *args):
        """Time to serialize data without initialization."""
        self.manager.serialize()
