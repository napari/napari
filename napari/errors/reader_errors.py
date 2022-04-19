from typing import List


class MultipleReaderError(RuntimeError):
    """Multiple readers are available for a path and none explicitly chosen.

    Thrown when the viewer model tries to open a file but multiple
    reader plugins are available that could claim it. User must
    make an explicit choice out of the available readers before opening
    file.

    Parameters
    ----------
    available_readers : List[str]
        list of available reader plugins for path
    pth: str
        file path for reading

    Attributes
    ----------
    available_readers : List[str]
        list of available reader plugins for path
    pth: str
        file path for reading
    """

    def __init__(self, available_readers: List[str], pth: str, *args: object):
        super().__init__(*args)
        self.available_plugins = available_readers
        self.pth = pth

    def __str__(self):
        return f"Multiple plugins found capable of reading {self.pth}. Select plugin from {self.available_plugins} and pass to reading function e.g. `viewer.open(..., plugin=...)`."


class ReaderPluginError(ValueError):
    """A reader plugin failed while trying to open a path.

    This error is thrown either when the only available plugin
    failed to read the path, or when the plugin associated with the
    path's file extension failed, or is unavailable.

    Parameters
    ----------
    reader_plugin : str
        plugin that was tried
    pth: str
        path the plugin tried to read

    Attributes
    ----------
    reader_plugin : str
        plugin that was tried
    pth: str
        path the plugin tried to read
    """

    def __init__(self, reader_plugin: str, pth: str, *args: object) -> None:
        super().__init__(*args)
        self.reader_plugin = reader_plugin
        self.pth = pth


class NoAvailableReaderError(ValueError):
    """No reader plugins are available to open the chosen file

    Parameters
    ----------
    pth: str
        file path for reading

    Attributes
    ----------
    pth: str
        file path for reading
    """

    def __init__(self, pth: str, *args: object) -> None:
        super().__init__(*args)
        self.pth = pth
