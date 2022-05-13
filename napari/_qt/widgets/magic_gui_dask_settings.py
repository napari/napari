from magicgui import magicgui

from .magic_gui_base_widget import BaseMagicSetting


@magicgui(auto_call=True, layout='horizontal', cache={'min': 0, 'max': 20})
def dask_settings(dask_enabled=True, cache=15.0) -> int:
    """Create magic gui function GUI with checkbox and spinbox for dask settings."""
    return {'enabled': dask_enabled, 'cache': cache}


class DaskSettings(BaseMagicSetting):
    """Class for use in json schema widget builder for dask settings."""

    MAGIC_GUI = dask_settings
