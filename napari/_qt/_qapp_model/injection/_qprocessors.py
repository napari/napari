"""Qt processors.

Non-Qt processors can be found in `napari/_app_model/injection/_processors.py`.
"""

from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

from magicgui.widgets import FunctionGui, Widget
from qtpy.QtWidgets import QWidget

from napari import viewer
from napari._app_model.injection._providers import _provide_viewer
from napari.utils.translations import trans


def _add_plugin_dock_widget(
    widget_name_tuple: Tuple[Union[FunctionGui, QWidget, Widget], str],
    viewer: Optional[viewer.Viewer] = None,
) -> None:
    if viewer is None:
        viewer = _provide_viewer()
    if viewer:
        widget, display_name = widget_name_tuple
        viewer.window.add_dock_widget(widget, name=display_name)
    else:
        raise RuntimeError(
            trans._(
                "No current `Viewer` found. Widgets cannot be opened in headless mode.",
                deferred=True,
            )
        )


QPROCESSORS: Dict[object, Callable] = {
    Optional[
        Tuple[Union[FunctionGui, QWidget, Widget], str]
    ]: _add_plugin_dock_widget,
}
