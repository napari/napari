"""ConsoleCommands class for IPython console and related table classes.
"""
from typing import List, Tuple

import humanize

from ..utils.chunk import chunk_loader, LayerInfo
from ..utils.text_color import text_color
from ..utils.text_table import TextTable


def _strong(string: str) -> str:
    return text_color(string, "cyan")


HELP_STR = f"""
{_strong("Available Commands:")}
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
        return self.info.data_type if self.info else "--"

    @property
    def num_loads(self):
        return self.info.num_loads if self.info else "--"

    @property
    def num_chunks(self):
        return self.info.num_chunks if self.info else "--"

    @property
    def total(self):
        if not self.info:
            return "--"
        # gnu=True gives the short "103M" or "92K" suffixes.
        return humanize.naturalsize(self.info.num_bytes, gnu=True)

    @property
    def avg_ms(self):
        if not self.info:
            return "--"
        ms = self.info.load_time_ms.average
        return f"{ms:.1f}"


class ListLayersTable:
    """Table showing the layers and their names, types and shapes.

    Parameters
    ----------
    layers : List[Layer]
        The layers to list in the table.

    Attributes
    ----------
    table : TextTable
        Formats our table for printing.
    """

    def __init__(self, layers):
        self.layers = layers
        self.table = TextTable(
            [
                "ID",
                "NAME",
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
        """Get the number of levels of the data."""
        if isinstance(data, list):
            return len(data)
        return 1

    def _add_row(self, i, layer):
        """Add one row to the layer list table."""
        layer_type = type(layer).__name__
        num_levels = self._get_num_levels(layer.data)
        shape_str = self._get_shape_str(layer.data)

        # Get info for this layer.
        info = chunk_loader.get_info(id(layer))

        # Displayer for the info, info could be None.
        disp = InfoDisplayer(info)

        self.table.add_row(
            [
                i,
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

    """

    def __init__(self, layer_id, layer):
        self.layer_id = layer_id
        self.layer = layer
        self.table = TextTable(["LEVEL", "SHAPE"])
        data = layer.data
        if isinstance(data, list):
            for i, level in enumerate(data):
                shape_str = level.shape if level.shape else "NONE"
                self.table.add_row([i, shape_str])

    def print(self):
        """Print the whole table."""

        self.table.print()


def _print_summary(items: List[Tuple[str, str]]):
    """Print a summary table with aligned headings.

    For example prints:

    Layer ID: 0
        Name: numbered slices
      Levels: 1
       Shape: (20, 1024, 1024, 3)

    Parameters
    ----------
        rows
    """
    width = max(len(heading) for heading, _ in items)
    for heading, value in items:
        aligned = f"{heading:>{width}}"
        print(f"{_strong(aligned)}: {value}")


class ConsoleCommands:
    """Command object for interactive use in the console.

    Usage:
        viewer.cmd.help
        viewer.cmd.list
        etc.
    """

    def __init__(self, layerlist):
        self.layers = layerlist

    def __repr__(self):
        return HELP_STR

    @property
    def help(self):
        print(HELP_STR)

    @property
    def list(self):
        """Print the current list of layers."""
        table = ListLayersTable(self.layers)
        table.print()

    def info(self, layer_id):
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
        try:
            layer = self.layers[layer_id]
        except IndexError:
            print(f"Invalid layer index: {layer_id}")
            return

        num_levels = len(layer.data) if layer.multiscale else 1

        # Common to both multi-scale and single-scale.
        summary = [
            ("Layer ID", layer_id),
            ("Name", layer.name),
            ("Levels", num_levels),
        ]

        if layer.multiscale:
            # Print summary and level table.
            _print_summary(summary)
            print("")  # blank line
            LevelsTable(layer_id, layer).print()
        else:
            # Print summary with shape, no level table.
            summary.append(("Shape", layer.data.shape))
            _print_summary(summary)
