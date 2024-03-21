from typing import List

from napari.types import PathLike


class MultipleReaderError(RuntimeError):
    """Multiple readers are available for paths and none explicitly chosen.

    Thrown when the viewer model tries to open files but multiple
    reader plugins are available that could claim them. User must
    make an explicit choice out of the available readers before opening
    files.

    Parameters
    ----------
    message: str
        error description
    available_readers : List[str]
        list of available reader plugins for path
    paths: List[str]
        file paths for reading

    Attributes
    ----------
    message: str
        error description
    available_readers : List[str]
        list of available reader plugins for path
    paths: List[str]
        file paths for reading
    """

    def __init__(
        self,
        message: str,
        available_readers: List[str],
        paths: List[PathLike],
        *args: object,
    ) -> None:
        super().__init__(message, *args)
        self.available_plugins = available_readers
        self.paths = paths


class ReaderPluginError(ValueError):
    """A reader plugin failed while trying to open paths.

    This error is thrown either when the only available plugin
    failed to read the paths, or when the plugin associated with the
    paths' file extension failed.

    Parameters
    ----------
    message: str
        error description
    reader_plugin : str
        plugin that was tried
    paths: List[str]
        file paths for reading

    Attributes
    ----------
    message: str
        error description
    reader_plugin : str
        plugin that was tried
    paths: List[str]
        file paths for reading
    """

    def __init__(
        self,
        message: str,
        reader_plugin: str,
        paths: List[PathLike],
        *args: object,
    ) -> None:
        super().__init__(message, *args)
        self.reader_plugin = reader_plugin
        self.paths = paths


class NoAvailableReaderError(ValueError):
    """No reader plugins are available to open the chosen file

    Parameters
    ----------
    message: str
        error description
    paths: List[str]
        file paths for reading

    Attributes
    ----------
    message: str
        error description
    paths: List[str]
        file paths for reading
    """

    def __init__(
        self, message: str, paths: List[PathLike], *args: object
    ) -> None:
        super().__init__(message, *args)
        self.paths = paths
