from __future__ import annotations

from typing import Any

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.layers.image import VispyImageLayer
from napari._vispy.layers.labels import VispyLabelsLayer
from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.layers.shapes import VispyShapesLayer
from napari._vispy.layers.surface import VispySurfaceLayer
from napari._vispy.layers.tracks import VispyTracksLayer
from napari._vispy.layers.vectors import VispyVectorsLayer
from napari._vispy.overlays.base import VispyBaseOverlay
from napari._vispy.overlays.bounding_box import VispyBoundingBoxOverlay
from napari._vispy.overlays.brush_circle import VispyBrushCircleOverlay
from napari._vispy.overlays.canvas_axes import VispyCanvasAxesOverlay
from napari._vispy.overlays.colorbar import VispyColorBarOverlay
from napari._vispy.overlays.crosshair import VispyCrosshairOverlay
from napari._vispy.overlays.grid_lines import VispyGridLinesOverlay
from napari._vispy.overlays.interaction_box import (
    VispySelectionBoxOverlay,
    VispyTransformBoxOverlay,
)
from napari._vispy.overlays.labels_brush_stroke import (
    VispyLabelsBrushStrokeOverlay,
)
from napari._vispy.overlays.labels_polygon import VispyLabelsPolygonOverlay
from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari._vispy.overlays.scene_axes import VispySceneAxesOverlay
from napari._vispy.overlays.text import (
    VispyCurrentSliceOverlay,
    VispyLayerNameOverlay,
    VispyTextOverlay,
)
from napari._vispy.overlays.zoom import VispyZoomOverlay
from napari.components.overlays import (
    BoundingBoxOverlay,
    BrushCircleOverlay,
    CanvasAxesOverlay,
    ColorBarOverlay,
    CrosshairOverlay,
    CurrentSliceOverlay,
    GridLinesOverlay,
    LabelsBrushStrokeOverlay,
    LabelsPolygonOverlay,
    LayerNameOverlay,
    Overlay,
    ScaleBarOverlay,
    SceneAxesOverlay,
    SelectionBoxOverlay,
    TextOverlay,
    TransformBoxOverlay,
    ZoomOverlay,
)
from napari.layers import (
    Image,
    Labels,
    Layer,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)

layer_to_visual: dict[type[Layer], type[VispyBaseLayer]] = {
    Image: VispyImageLayer,
    Labels: VispyLabelsLayer,
    Points: VispyPointsLayer,
    Shapes: VispyShapesLayer,
    Surface: VispySurfaceLayer,
    Vectors: VispyVectorsLayer,
    Tracks: VispyTracksLayer,
}


overlay_to_visual: dict[type[Overlay], type[VispyBaseOverlay]] = {
    ScaleBarOverlay: VispyScaleBarOverlay,
    TextOverlay: VispyTextOverlay,
    GridLinesOverlay: VispyGridLinesOverlay,
    SceneAxesOverlay: VispySceneAxesOverlay,
    CanvasAxesOverlay: VispyCanvasAxesOverlay,
    BoundingBoxOverlay: VispyBoundingBoxOverlay,
    TransformBoxOverlay: VispyTransformBoxOverlay,
    SelectionBoxOverlay: VispySelectionBoxOverlay,
    BrushCircleOverlay: VispyBrushCircleOverlay,
    LabelsPolygonOverlay: VispyLabelsPolygonOverlay,
    CrosshairOverlay: VispyCrosshairOverlay,
    LabelsBrushStrokeOverlay: VispyLabelsBrushStrokeOverlay,
    ZoomOverlay: VispyZoomOverlay,
    LayerNameOverlay: VispyLayerNameOverlay,
    CurrentSliceOverlay: VispyCurrentSliceOverlay,
    ColorBarOverlay: VispyColorBarOverlay,
}


def create_vispy_layer(
    layer: Layer, *args: Any, **kwargs: Any
) -> VispyBaseLayer:
    """Create vispy visual for a layer based on its layer type.

    Parameters
    ----------
    layer : napari.layers._base_layer.Layer
        Layer that needs its property widget created.

    Returns
    -------
    visual : VispyBaseLayer
        Vispy layer
    """
    # find the closest parent class, to maintain behaviour from #2757
    for cls in layer.__class__.mro():
        if cls in layer_to_visual:
            return layer_to_visual[cls](layer, *args, **kwargs)

    raise TypeError(
        f'Could not find VispyLayer for layer of type {type(layer)}'
    )


def create_vispy_overlay(overlay: Overlay, **kwargs) -> VispyBaseOverlay:
    """
    Create vispy visual for Overlay based on its type.

    Parameters
    ----------
    overlay : napari.components.overlays.VispyBaseOverlay
        The overlay to create a visual for.

    Returns
    -------
    visual : VispyBaseOverlay
        Vispy overlay
    """
    for cls in overlay.__class__.mro():
        if cls in overlay_to_visual:
            return overlay_to_visual[cls](overlay=overlay, **kwargs)

    raise TypeError(
        f'Could not find VispyOverlay for overlay of type {type(overlay)}'
    )
