class PluginError(Exception):
    def __init__(
        self, message: str, plugin_name: str, plugin_module: str
    ) -> None:
        super().__init__(message)
        self.plugin_name = plugin_name
        self.plugin_module = plugin_module


class PluginImportError(PluginError, ImportError):
    """Raised when a plugin fails to import."""

    def __init__(self, plugin_name: str, plugin_module: str) -> None:
        msg = f'Failed to import plugin: "{plugin_name}"'
        super().__init__(msg, plugin_name, plugin_module)


class PluginRegistrationError(PluginError):
    """Raised when a plugin fails to register with pluggy."""

    def __init__(self, plugin_name: str, plugin_module: str) -> None:
        msg = f'Failed to register plugin: "{plugin_name}"'
        super().__init__(msg, plugin_name, plugin_module)
