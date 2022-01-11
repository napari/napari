from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from ....utils.events.custom_types import Array
from ....utils.translations import trans
from ...utils.layer_utils import _features_to_properties


class BaseTrackManager(ABC):
    """Base class to manage to specify the expected API of the TrackManager to manipulate
    the track data and simplify interactions with the Tracks layer.
    """

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """array (N, D+1): Coordinates for N points in D+1 dimensions."""

    @data.setter
    @abstractmethod
    def data(self, data: Union[list, np.ndarray]):
        """set the vertex data and build the vispy arrays for display"""

    @property
    @abstractmethod
    def features(self):
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1].
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1]: https://data-apis.org/dataframe-protocol/latest/API.html
        """

    @features.setter
    @abstractmethod
    def features(
        self,
        features: Union[Dict[str, np.ndarray], pd.DataFrame],
    ) -> None:
        pass

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}: Properties for each track."""
        return _features_to_properties(self.features)

    @properties.setter
    def properties(self, properties: Dict[str, Array]):
        """set track properties"""
        self.features = properties

    @property
    @abstractmethod
    def graph(self) -> Dict[int, Union[int, List[int]]]:
        """dict {int: list}: Graph representing associations between tracks."""

    @graph.setter
    @abstractmethod
    def graph(self, graph: Dict[int, Union[int, List[int]]]):
        """set the track graph"""

    @property
    def track_ids(self):
        """return the track identifiers"""
        return self.data[:, 0].astype(np.uint32)

    @property
    def unique_track_ids(self):
        """return the unique track identifiers"""
        return np.unique(self.track_ids)

    def __len__(self):
        """return the number of tracks"""
        return len(self.unique_track_ids) if self.data is not None else 0

    @abstractmethod
    def build_tracks(self):
        """build the tracks"""

    @abstractmethod
    def build_graph(self):
        """build the track graph"""

    def vertex_properties(self, color_by: str) -> np.ndarray:
        """return the properties of tracks by vertex"""

        if color_by not in self.properties:
            raise ValueError(
                trans._(
                    'Property {color_by} not found',
                    deferred=True,
                    color_by=color_by,
                )
            )

        return self.properties[color_by]

    @abstractmethod
    def get_value(self, coords):
        """lookup the ID of the nearest track node"""

    @property
    def ndim(self) -> int:
        """Determine number of spatiotemporal dimensions of the layer."""
        return self.data.shape[1] - 1

    @property
    def max_time(self) -> int:
        """Determine the maximum timestamp of the dataset"""
        return int(np.max(self.track_times))

    @property
    @abstractmethod
    def track_vertices(self) -> np.ndarray:
        """return the track vertices"""

    @property
    @abstractmethod
    def track_connex(self) -> np.ndarray:
        """vertex connections for drawing track lines"""

    @property
    @abstractmethod
    def graph_vertices(self) -> np.ndarray:
        """return the graph vertices"""

    @property
    @abstractmethod
    def graph_connex(self):
        """vertex connections for drawing the graph"""

    @property
    @abstractmethod
    def track_times(self) -> np.ndarray:
        """time points associated with each track vertex"""
        return self.track_vertices[:, 0]

    @property
    @abstractmethod
    def graph_times(self) -> np.ndarray:
        """time points associated with each graph vertex"""
        if self.graph_vertices is not None:
            return self.graph_vertices[:, 0]
        return None

    @abstractmethod
    def track_labels(self, current_time: int) -> tuple:
        """return track labels at the current time"""

    def view_graph(self, *args, **kwargs) -> np.ndarray:
        return self.graph

    def view_graph_vertices(self, *args, **kwargs) -> np.ndarray:
        return self.graph_vertices

    def view_graph_connex(self, *args, **kwargs) -> np.ndarray:
        return self.graph_connex

    def view_graph_times(self, *args, **kwargs) -> np.ndarray:
        if self.graph_vertices is not None:
            return self.graph_vertices[:, 0]
        return None

    def view_track_vertices(self, *args, **kwargs) -> np.ndarray:
        return self.track_vertices

    def view_track_ids(self, *args, **kwargs) -> np.ndarray:
        return self.track_ids

    def view_track_connex(self, *args, **kwargs) -> np.ndarray:
        return self.track_connex

    def view_track_times(self, *args, **kwargs) -> np.ndarray:
        return self.track_vertices[:, 0]

    def view_features(self, *args, **kwargs) -> np.ndarray:
        return self.features
