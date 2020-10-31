"""
All napari hook specifications for pluggable functionality are defined here.

A *hook specification* is a function signature (with documentation) that
declares an API that plugin developers must adhere to when providing hook
implementations.  *Hook implementations* provided by plugins (and internally by
napari) will then be invoked in various places throughout the code base.

When implementing a hook specification, pay particular attention to the number
and types of the arguments in the specification signature, as well as the
expected return type.

To allow for hook specifications to evolve over the lifetime of napari,
hook implementations may accept *fewer* arguments than defined in the
specification. (This allows for extending existing hook arguments without
breaking existing implementations). However, implementations must not require
*more* arguments than defined in the spec.

For more general background on the plugin hook calling mechanism, see the
`napari-plugin-manager documentation
<https://napari-plugin-engine.readthedocs.io/en/latest/>`_.

.. NOTE::
    Hook specifications are a feature borrowed from `pluggy
    <https://pluggy.readthedocs.io/en/latest/#specs>`_. In the `pluggy
    documentation <https://pluggy.readthedocs.io/en/latest/>`_, hook
    specification marker instances are named ``hookspec`` by convention, and
    hook implementation marker instances are named ``hookimpl``.  The
    convention in napari is to name them more explicitly:
    ``napari_hook_specification`` and ``napari_hook_implementation``,
    respectively.
"""

# These hook specifications also serve as the API reference for plugin
# developers, so comprehensive documentation with complete type annotations is
# imperative!

from typing import List, Tuple

from napari_plugin_engine import napari_hook_specification
from qtpy.QtWidgets import QWidget

# -------------------------------------------------------------------------- #
#                                 GUI Hooks                                  #
# -------------------------------------------------------------------------- #


@napari_hook_specification()
def napari_experimental_provide_dock_widget() -> List[Tuple[QWidget, dict]]:
    """Provide dock widgets that can be added to the viewer.

    Returns
    -------
    dock_widgets : list
        List of 2-tuples, where each tuple has a widget and dictionary of
        keyword arguments for the viewer.window.add_dock_widget method.
    """
