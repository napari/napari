"""LoaderCommands class and helpers.
"""
from typing import List

import dask.array as da

from ._humanize import naturalsize
from ..layers.base import Layer
from ._tables import print_property_table, RowTable
from ._utils import highlight
from ..utils.chunk import async_config, chunk_loader, LayerInfo


HELP_STR = f"""
{highlight("Available Commands:")}
cmd.help
cmd.list
cmd.info(layer_id)
"""


class InfoDisplayer:
    """Display LayerInfo values nicely for the table.

    Most values display as "--" if we don't have an info. TODO_ASYNC: is
    there a nicer way to accomplish this? Have '--' as default for everything?

    Some values require a bit of computation or formatting.

    Parameters
    ----------
    layer_info : LayerInfo
        The LayerInfo to display.
    """

    def __init__(self, layer_info: LayerInfo):
        self.info = layer_info

    @property
    def data_type(self):
        return self.info.data_type

    @property
    def num_loads(self):
        return self.info.num_loads

    @property
    def num_chunks(self):
        return self.info.num_chunks

    @property
    def total(self):
        # gnu=True gives the short "103M" or "92K" suffixes.
        return naturalsize(self.info.num_bytes, gnu=True)

    @property
    def avg_ms(self):
        ms = self.info.load_time_ms.average
        return f"{ms:.1f}"


class NoInfoDisplayer:
    """When we have no LayerInfo every field is just blank."""

    def __getattr__(self, name):
        return "--"


def _get_type_str(data) -> str:
    """Get human readable name for the data's type.

    Returns
    -------
    str
        A string like "ndarray" or "dask".
    """
    if isinstance(data, list):
        if len(data) == 0:
            return "EMPTY"
        else:
            # Recursively get the type string of the zeroth level.
            return _get_type_str(data[0])

    if type(data) == da.Array:
        # Special case this because otherwise data_type.__name__
        # below would just return "Array".
        return "dask"

    # For class numpy.ndarray this returns "ndarray"
    return type(data).__name__


def _get_size_str(data) -> str:
    """Return human readable size.

    Parameters
    ----------
    data
        Layer's data.

    Returns
    -------
    str
        A string size like "24.2G".
    """
    if isinstance(data, list):
        nbytes = sum(level.nbytes for level in data)
    else:
        nbytes = data.nbytes
    return naturalsize(nbytes, gnu=True)


class ChunkLoaderTable:
    """Table showing information about each layer.

    Parameters
    ----------
    layers : List[Layer]
        The layers to list in the table.

    Attributes
    ----------
    table : TextTable
        Formats our table for printing.
    """

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.table = RowTable(
            [
                "ID",
                {"name": "NAME", "align": "left"},
                "LAYER",
                "DATA",
                "LEVELS",
                "LOADS",
                "CHUNKS",
                "TOTAL",
                "AVG (ms)",
                "SHAPE",
            ]
        )
        for i, layer in enumerate(self.layers):
            self._add_row(i, layer)

    def _get_shape_str(self, data):
        """Get shape string for the data.

        Either "NONE" or a tuple like "(10, 100, 100)".
        """
        if isinstance(data, list):
            if len(data) == 0:
                return "NONE"  # Shape layer is empty list?
            else:
                return f"{data[0].shape}"  # Multi-scale
        else:
            return str(data.shape)

    @staticmethod
    def _get_num_levels(data) -> int:
        """Get the number of levels of the data.

        Parameters
        ----------
        data
            Layer data.

        Returns
        -------
        int
            The number of levels of the data.
        """
        if isinstance(data, list):
            return len(data)
        return 1

    def _add_row(self, index: int, layer: Layer) -> int:
        """Add row describing one layer.

        Parameters
        ----------
        layer_id : int
            The layer id (from view.cmd.layers).
        layer : Layer
            The layer itself.
        """
        layer_type = type(layer).__name__
        num_levels = self._get_num_levels(layer.data)
        shape_str = self._get_shape_str(layer.data)

        # Get LayerInfo and use the InfoDisplayer.
        info = chunk_loader.get_info(id(layer))
        disp = InfoDisplayer(info) if info is not None else NoInfoDisplayer()

        self.table.add_row(
            [
                index,
                layer.name,
                layer_type,
                disp.data_type,
                num_levels,
                disp.num_loads,
                disp.num_chunks,
                disp.total,
                disp.avg_ms,
                shape_str,
            ]
        )

    def print(self):
        """Print the whole table."""
        self.table.print()


class LevelsTable:
    """Table showing the levels in a single layer.

    Parameters
    ----------
    layer_id : int
        The ID of this layer.
    layer : Layer
        Show the levels of this layer.
    """

    def __init__(self, layer_id: int, layer):
        self.layer_id = layer_id
        self.layer = layer
        self.table = RowTable(["LEVEL", "SHAPE", "TOTAL"])
        self.table = RowTable(
            ["LEVEL", {"name": "SHAPE", "align": "left"}, "TOTAL"]
        )
        data = layer.data
        if isinstance(data, list):
            for i, level in enumerate(data):
                shape_str = level.shape if level.shape else "NONE"
                size_str = naturalsize(level.nbytes, gnu=True)
                self.table.add_row([i, shape_str, size_str])

    def print(self):
        """Print the whole table."""

        self.table.print()


class LoaderCommands:
    """Layer related commands for the CommandProcessor.

    Parameters
    ----------
    layerlist : List[Layer]
        The current list of layers.
    """

    def __init__(self, layerlist: List[Layer]):
        self.layerlist = layerlist

    @property
    def loader_config(self):
        """Print the current list of layers."""
        src = async_config
        config = [
            ('synchronous', src.synchronous),
            ('num_workers', src.num_workers),
            ('log_path', src.log_path),
            ('use_processes', src.use_processes),
            ('delay_seconds', src.delay_seconds),
            ('load_seconds', src.load_seconds),
        ]
        print_property_table(config)

    @property
    def loader(self):
        """Print the current list of layers."""
        ChunkLoaderTable(self.layerlist).print()
