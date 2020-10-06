"""Experimental CommandProcessor class.
"""


class CommandProcessor:
    def __init__(self, layers):
        self.layers = layers

    @property
    def loader(self):
        from .chunk._commands import LoaderCommands

        return LoaderCommands(self.layers)
