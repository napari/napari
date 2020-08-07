"""ConsoleProcessor class for our IPython console.
"""
from ._utils import highlight
from ._layers import LayerCommands

HELP_STR = f"""
{highlight("Available Commands:")}
cmd.help
cmd.layers
cmd.levels(layer_id)
"""


class CommandProcessor:
    """Command processor designed for interactive use in the IPython console.

    Type "viewer.cmd.help" in Python for valid commands.
    """

    def __init__(self, layerlist):
        self.layer_commands = LayerCommands(layerlist)

    def __repr__(self):
        return HELP_STR

    @property
    def help(self):
        print(HELP_STR)

    @property
    def layers(self):
        """Print a table with the current layers."""
        return self.layer_commands.layers

    def levels(self, layer_id):
        """Print information about a single layer."""
        return self.layer_commands.levels(layer_id)
