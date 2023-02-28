"""LoaderCommands class and helpers.
"""
from typing import List

from napari._vendor.experimental.humanize.src.humanize import naturalsize
from napari.components.experimental.chunk._commands._tables import (
    RowTable,
    print_property_table,
)
from napari.components.experimental.chunk._commands._utils import highlight
from napari.components.experimental.chunk._info import LayerInfo, LoadType
from napari.components.experimental.chunk._loader import chunk_loader
from napari.layers.base import Layer
from napari.layers.image import Image
from napari.utils.config import octree_config

LOAD_TYPE_STR = {
    LoadType.AUTO: "auto",
    LoadType.SYNC: "sync",
    LoadType.ASYNC: "async",
}

HELP_STR = f"""
{highlight("Available Commands:")}
loader.help
loader.cache
loader.config
loader.layers
loader.levels(index)
loader.loads(index)
loader.set_default(index)
loader.set_sync(index)
loader.set_async(index)
"""


def format_bytes(num_bytes):
    """Return formatted string like K, M, G.

    The gnu=True flag produces GNU-style single letter suffixes which
    are more compact then KiB, MiB, GiB.
    """
    return naturalsize(num_bytes, gnu=True)


class InfoDisplayer:
    """Display LayerInfo values nicely for the table.

    This mainly exist so we can have NoInfoDisplay which displays "--"
    for all the values. Seemed like the easiest way to handle the
    case when we have info and the case when we don't.

    Parameters
    ----------
    layer_info : LayerInfo
        The LayerInfo to display.
    """

    def __init__(self, info: LayerInfo) -> None:
        self.info = info
        stats = info.stats
        counts = stats.counts

        self.data_type = "???"  # We need to add this back...
        self.num_loads = counts.loads
        self.num_chunks = counts.chunks
        self.sync = LOAD_TYPE_STR[self.info.load_type]
        self.total = format_bytes(counts.bytes)
        self.avg_ms = f"{stats.window_ms.average:.1f}"
        self.mbits = f"{stats.mbits:.1f}"
        self.load_str = stats.recent_load_str


class NoInfoDisplayer:
    """When we have no LayerInfo every field is just blank."""

    def __getattr__(self, name):
        return "--"


def _get_shape_str(layer):
    """Get shape string for the data.

    Either "NONE" or a tuple like "(10, 100, 100)".
    """
    # We only care about Image/Labels layers for now.
    if not isinstance(layer, Image):
        return "--"

    data = layer.data
    if isinstance(data, list):
        if len(data) == 0:
            return "NONE"  # Shape layer is empty list?
        return f"{data[0].shape}"  # Multi-scale

    # Not a list.
    return str(data.shape)


class ChunkLoaderLayers:
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

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self.table = RowTable(
            [
                "ID",
                "MODE",
                "LOADS",
                {"name": "NAME", "align": "left"},
                "LAYER",
                "DATA",
                "LEVELS",
                "LOADS",
                "CHUNKS",
                "TOTAL",
                "AVG (ms)",
                "MBIT/s",
                "SHAPE",
            ]
        )
        for i, layer in enumerate(self.layers):
            self._add_row(i, layer)

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
        index : int
            The layer id (from view.cmd.layers).
        layer : Layer
            The layer itself.
        """
        layer_type = type(layer).__name__
        num_levels = self._get_num_levels(layer.data)
        shape_str = _get_shape_str(layer)

        # Use InfoDisplayer to display LayerInfo
        info = chunk_loader.get_info(id(layer))
        disp = InfoDisplayer(info) if info is not None else NoInfoDisplayer()

        self.table.add_row(
            [
                index,
                disp.sync,
                disp.load_str,
                layer.name,
                layer_type,
                disp.data_type,
                num_levels,
                disp.num_loads,
                disp.num_chunks,
                disp.total,
                disp.avg_ms,
                disp.mbits,
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

    def __init__(self, layer) -> None:
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

    def __init__(self, layerlist: List[Layer]) -> None:
        self.layerlist = layerlist

    def __repr__(self):
        return HELP_STR

    @property
    def help(self):
        """The help message."""
        print(HELP_STR)

    @property
    def config(self):
        """Print the current list of layers."""
        config = octree_config['loader']
        config = [
            ('log_path', config['log_path']),
            ('synchronous', config['synchronous']),
            ('num_workers', config['num_workers']),
            ('use_processes', config['use_processes']),
            ('auto_sync_ms', config['auto_sync_ms']),
            ('delay_queue_ms', config['delay_queue_ms']),
        ]
        print_property_table(config)

    @property
    def cache(self):
        """The cache status."""
        chunk_cache = chunk_loader.cache
        cur_str = format_bytes(chunk_cache.chunks.currsize)
        max_str = format_bytes(chunk_cache.chunks.maxsize)
        table = [
            ('enabled', chunk_cache.enabled),
            ('currsize', cur_str),
            ('maxsize', max_str),
        ]
        print_property_table(table)

    @property
    def layers(self):
        """Print the current list of layers."""
        ChunkLoaderLayers(self.layerlist).print()

    def _get_layer(self, layer_index) -> Layer:
        try:
            return self.layerlist[layer_index]
        except KeyError:
            print(f"Layer index {layer_index} is invalid.")
            return None

    def _get_layer_info(self, layer_index) -> LayerInfo:
        """Return the LayerInfo at this index."""
        layer = self._get_layer(layer_index)

        if layer is None:
            return None

        layer_id = id(layer)
        info = chunk_loader.get_info(layer_id)

        if info is None:
            print(f"Layer index {layer_index} has no LayerInfo.")
            return None

        return info

    def loads(self, layer_index: int) -> None:
        """Print recent loads for this layer.

        Attributes
        ----------
        layer_index : int
            The index from the viewer.cmd.loader table.
        """
        info = self._get_layer_info(layer_index)

        if info is None:
            return

        table = RowTable(["INDEX", "TYPE", "SIZE", "DURATION (ms)", "Mbit/s"])
        for i, load in enumerate(info.recent_loads):
            load_str = "sync" if load.sync else "async"
            duration_str = f"{load.duration_ms:.1f}"
            mbits_str = f"{load.mbits:.1f}"
            table.add_row(
                (i, load_str, load.num_bytes, duration_str, mbits_str)
            )

        table.print()

    def _set_load_type(self, index, load_type) -> None:
        """Set this layer to this load type."""
        info = self._get_layer_info(index)
        if info is not None:
            info.load_type = load_type

    def set_sync(self, index) -> None:
        """Set this layer to sync loading."""
        self._set_load_type(index, LoadType.SYNC)

    def set_async(self, index) -> None:
        """Set this layer to async loading."""
        self._set_load_type(index, LoadType.ASYNC)

    def set_auto(self, index) -> None:
        """Set this layer to auto loading."""
        self._set_load_type(index, LoadType.AUTO)

    def levels(self, layer_index: int) -> None:
        """Print information on a single layer.
        Prints summary and if multiscale prints a table of the levels:
        Layer ID: 0
            Name: LaminB1
          Levels: 2
        LEVEL  SHAPE
        0      (1, 236, 275, 271)
        1      (1, 236, 137, 135)
        Parameters
        ----------
        layer_id : int
            ConsoleCommand's id for the layer.
        """
        layer = self._get_layer(layer_index)
        if layer is None:
            return

        num_levels = len(layer.data) if layer.multiscale else 1

        # Common to both multi-scale and single-scale.
        summary = [
            ("Layer ID", layer_index),
            ("Name", layer.name),
            ("Levels", num_levels),
        ]

        if layer.multiscale:
            # Print summary and level table.
            print_property_table(summary)
            print("")  # blank line
            LevelsTable(layer).print()
        else:
            # Print summary with shape, no level table.
            summary.append(("Shape", layer.data.shape))
            print_property_table(summary)
