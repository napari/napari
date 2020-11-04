"""ExperimentalNamespace and CommandProcessor classes.
"""
from .chunk._commands._utils import highlight

HELP_STR = f"""
{highlight("Available Commands:")}
experimental.cmds.loader
"""


class CommandProcessor:
    def __init__(self, layers):
        self.layers = layers

    @property
    def loader(self):
        from .chunk._commands import LoaderCommands

        return LoaderCommands(self.layers)

    def __repr__(self):
        return "Available Commands:\nexperimental.cmds.loader"


class ExperimentalNamespace:
    def __init__(self, layers):
        self.layers = layers

    @property
    def cmds(self):
        return CommandProcessor(self.layers)

    def __repr__(self):
        return "Available Commands:\nexperimental.cmds.loader"
