from __future__ import annotations

from typing import TYPE_CHECKING, Union

from app_model.expressions import ContextKey

from napari._app_model.context._context_keys import ContextNamespace
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.viewer import Viewer, ViewerModel

    ViewerLike = Union[Viewer, ViewerModel]


def _is_viewer_3d(viewer: ViewerLike) -> bool:
    return viewer.dims.ndisplay == 3


def _is_viewer_grid_enabled(viewer: ViewerLike) -> bool:
    return viewer.grid.enabled


class ViewerContextKeys(ContextNamespace['ViewerLike']):
    is_viewer_3d = ContextKey(
        False,
        trans._('True when the viewer is in 3D mode.'),
        _is_viewer_3d,
    )
    is_viewer_grid_enabled = ContextKey(
        False,
        trans._('True when the viewer grid is enabled.'),
        _is_viewer_grid_enabled,
    )

    def update_from_source(self, source: ViewerLike) -> None:
        """Trigger an update of all "getter" functions in this namespace."""
        for k, get in self._getters.items():
            setattr(self, k, get(source))
