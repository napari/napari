import os
import sys
from inspect import isclass
from typing import Callable, Dict, Sequence, Tuple, Type, Union

from napari_plugin_engine import PluginManager
from qtpy.QtWidgets import QAction, QWidget

from ..utils._appdirs import user_site_packages
from ..utils.misc import camel_to_spaces, is_sequence, running_as_bundled_app
from . import _builtins, hook_specifications

if sys.platform.startswith('linux') and running_as_bundled_app():
    sys.path.append(user_site_packages())


# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager(
    'napari', discover_entry_point='napari.plugin', discover_prefix='napari_'
)
with plugin_manager.discovery_blocked():
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')


dock_widgets: Dict[str, Type[QWidget]] = dict()
functions: Dict[str, Type[Tuple[Callable, Dict, Dict]]] = dict()


def register_dock_widget(cls: Union[Type[QWidget], Sequence[Type[QWidget]]]):
    for _cls in cls if is_sequence(cls) else [cls]:
        if not isclass(_cls) and issubclass(_cls, QWidget):
            # what to do here?
            continue
        name = getattr(
            _cls, 'napari_menu_name', camel_to_spaces(_cls.__name__)
        )
        if name in dock_widgets:
            # duplicate menu names... what to do here? Can this be namespaced by plugin name?
            continue
        dock_widgets[name] = _cls


def register_function(func):
    for _func in func if is_sequence(func) else [func]:
        # Add something to check cls is right type ....
        name = _func[0].__name__
        if name in functions:
            # duplicate menu names... what to do here? Can this be namespaced by plugin name?
            continue
        functions[name] = _func


plugin_manager.hook.napari_experimental_provide_dock_widget.call_historic(
    result_callback=register_dock_widget
)


plugin_manager.hook.napari_experimental_provide_function.call_historic(
    result_callback=register_function
)


__all__ = [
    "PluginManager",
    "plugin_manager",
]
