class MultiplePluginError(RuntimeError):
    pass


class ReaderPluginError(ValueError):
    def __init__(self, reader_plugin, *args: object) -> None:
        super().__init__(*args)
        self._reader_plugin = reader_plugin
