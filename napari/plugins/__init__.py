import os
import sys
from inspect import isclass
from typing import Dict, Sequence, Type, Union

from napari_plugin_engine import PluginManager
from qtpy.QtWidgets import QAction, QWidget

from ..utils._appdirs import user_site_packages
from ..utils.misc import camel_to_spaces, is_sequence, running_as_bundled_app
from . import _builtins, hook_specifications

if sys.platform.startswith('linux') and running_as_bundled_app():
    sys.path.append(user_site_packages())


if os.name == 'nt':
    # This is where plugins will be in bundled apps on windows
    exe_dir = os.path.dirname(sys.executable)
    winlib = os.path.join(exe_dir, "Lib", "site-packages")
    sys.path.append(winlib)

# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager(
    'napari', discover_entry_point='napari.plugin', discover_prefix='napari_'
)
with plugin_manager.discovery_blocked():
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')


dock_widgets: Dict[str, Type[QWidget]] = dict()


def register_dock_widget(cls: Union[Type[QWidget], Sequence[Type[QWidget]]]):
    for _cls in cls if is_sequence(cls) else [cls]:
        if not isclass(_cls) and issubclass(_cls, QWidget):
            # what to do here?
            continue
        name = getattr(
            _cls, 'napari_menu_name', camel_to_spaces(_cls.__name__)
        )
        if name in dock_widgets:
            # duplicate menu names... what to do here?
            continue
        dock_widgets[name] = _cls


plugin_manager.hook.napari_experimental_provide_dock_widget.call_historic(
    result_callback=register_dock_widget
)


__all__ = [
    "PluginManager",
    "plugin_manager",
]
