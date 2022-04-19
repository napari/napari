import os
from typing import List

from ..utils.translations import trans


class MultipleReaderError(RuntimeError):
    """Multiple readers are available for paths and none explicitly chosen.

    Thrown when the viewer model tries to open files but multiple
    reader plugins are available that could claim them. User must
    make an explicit choice out of the available readers before opening
    files.

    Parameters
    ----------
    available_readers : List[str]
        list of available reader plugins for path
    paths: List[str]
        file paths for reading

    Attributes
    ----------
    available_readers : List[str]
        list of available reader plugins for path
    paths: List[str]
        file paths for reading
    """

    def __init__(
        self, available_readers: List[str], paths: List[str], *args: object
    ):
        super().__init__(*args)
        self.available_plugins = available_readers
        self.paths = paths

    def __str__(self):
        return trans._(
            "Multiple plugins found capable of reading {path_message}. Select plugin from {plugins} and pass to reading function e.g. `viewer.open(..., plugin=...)`.",
            path_message=f"[{self.paths[0]}, ...]"
            if len(self.paths) > 1
            else self.paths[0],
            plugins=self.available_plugins,
        )


class ReaderPluginError(ValueError):
    """A reader plugin failed while trying to open paths.

    This error is thrown either when the only available plugin
    failed to read the paths, or when the plugin associated with the
    paths' file extension failed.

    Parameters
    ----------
    reader_plugin : str
        plugin that was tried
    paths: List[str]
        file paths for reading

    Attributes
    ----------
    reader_plugin : str
        plugin that was tried
    paths: List[str]
        file paths for reading
    """

    def __init__(
        self, reader_plugin: str, paths: List[str], *args: object
    ) -> None:
        super().__init__(*args)
        self.reader_plugin = reader_plugin
        self.paths = paths


class MissingAssociatedReaderError(RuntimeError):
    """The reader plugin associated with paths' extension is not available.

    Parameters
    ----------
    reader_plugin : str
        plugin that was tried
    paths: List[str]
        file paths for reading

    Attributes
    ----------
    reader_plugin : str
        plugin that was tried
    paths: List[str]
        file paths for reading
    """

    def __init__(
        self, reader_plugin: str, paths: List[str], *args: object
    ) -> None:
        super().__init__(*args)
        self.reader_plugin = reader_plugin
        self.paths = paths

    def __str__(self) -> str:
        return trans._(
            "Can't find {plugin} plugin associated with {extension} files.",
            plugin=self.reader_plugin,
            extension=os.path.splitext(self.paths[0])[1],
        )


class NoAvailableReaderError(ValueError):
    """No reader plugins are available to open the chosen file

    Parameters
    ----------
    paths: List[str]
        file paths for reading

    Attributes
    ----------
    paths: List[str]
        file paths for reading
    """

    def __init__(self, paths: List[str], *args: object) -> None:
        super().__init__(*args)
        self.paths = paths
