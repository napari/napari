from magicgui import magic_factory

from napari._qt.widgets.magic_gui_base_widget import BaseMagicSetting


@magic_factory(
    auto_call=True, layout='horizontal', cache={'min': 0, 'max': 20}
)
def dask_settings(dask_enabled=True, cache=15.0) -> dict:
    """Create magic gui function GUI with checkbox and spinbox for dask settings."""
    return {'enabled': dask_enabled, 'cache': cache}


class MagicDaskSettingsWidget(BaseMagicSetting):
    """Class for use in json schema widget builder for dask settings."""

    def get_mgui(self):
        return dask_settings()
