from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari")
except PackageNotFoundError:
    __version__ = 'unknown'
