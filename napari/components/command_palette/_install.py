from napari.components.command_palette._api import get_palette
from napari.components.command_palette._components import get_storage

APPNAME = "napari"
palette = get_palette(APPNAME)
storage = get_storage(APPNAME)
storage.mark_getter("viewer", lambda viewer: viewer)
