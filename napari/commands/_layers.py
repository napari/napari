"""LayerCommands and supporting classes.
"""
from typing import List, Tuple

import dask.array as da
import humanize

from ._tables import print_property_table, RowTable
from ._utils import highlight


HELP_STR = f"""
{highlight("Available Commands:")}
cmd.help
cmd.list
cmd.info(layer_id)
"""


def _get_type_str(data) -> str:
    """Get human readable name for the data's type.

    Returns
    -------
    str
        A string like "ndarray" or "dask".
    """
    data_type = type(data)

    if data_type == list:
        if len(data) == 0:
            return "EMPTY"
        else:
            # Recursively get the type string of the zeroth level.
            return _get_type_str(data[0])

    if data_type == da.Array:
        # Special case this because otherwise data_type.__name__
        # below would just return "Array".
        return "dask"

    # For class numpy.ndarray this returns "ndarray"
    return data_type.__name__


def _get_size_str(data) -> str:
    """Return human readable size like 24.2G."""
    if isinstance(data, list):
        nbytes = sum(level.nbytes for level in data)
    else:
        nbytes = data.nbytes
    return humanize.naturalsize(nbytes, gnu=True)


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
        self.table = RowTable(
            ["ID", "NAME", "LAYER", "DATA", "LEVELS", "SHAPE", "TOTAL"]
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
        """Add row describing one layer."""
        layer_type = type(layer).__name__
        num_levels = self._get_num_levels(layer.data)
        shape_str = self._get_shape_str(layer.data)
        data_type = _get_type_str(layer.data)
        size_str = _get_size_str(layer.data)

        self.table.add_row(
            [
                i,
                layer.name,
                layer_type,
                data_type,
                num_levels,
                shape_str,
                size_str,
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
        self.table = RowTable(["LEVEL", "SHAPE", "TOTAL"])
        self.table = RowTable(
            [{"name": "LEVEL", "align": "left"}, "SHAPE", "TOTAL"]
        )
        data = layer.data
        if isinstance(data, list):
            for i, level in enumerate(data):
                shape_str = level.shape if level.shape else "NONE"
                size_str = humanize.naturalsize(level.nbytes, gnu=True)
                self.table.add_row([i, shape_str, size_str])

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
        print(f"{highlight(aligned)}: {value}")


class LayerCommands:
    """Layer related commands for the CommandProcessor.
    """

    def __init__(self, layerlist):
        self.layerlist = layerlist

    @property
    def layers(self):
        """Print the current list of layers."""
        ListLayersTable(self.layerlist).print()

    def layer_info(self, layer_id):
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
            layer = self.layerlist[layer_id]
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
            print_property_table(summary)
            print("")  # blank line
            LevelsTable(layer_id, layer).print()
        else:
            # Print summary with shape, no level table.
            summary.append(("Shape", layer.data.shape))
            print_property_table(summary)
