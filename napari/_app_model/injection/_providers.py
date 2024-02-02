from typing import Optional

from napari import components, layers, viewer


def _provide_viewer() -> Optional[viewer.Viewer]:
    return viewer.current_viewer()


def _provide_active_layer() -> Optional[layers.Layer]:
    return v.layers.selection.active if (v := _provide_viewer()) else None


def _provide_active_layer_list() -> Optional[components.LayerList]:
    return v.layers if (v := _provide_viewer()) else None


# fmt: off
def _provide_active_image() -> Optional[layers.Image]:
    return l if (l := _provide_active_layer()) and type(l) == layers.Image else None  #noqa: E741


def _provide_active_labels() -> Optional[layers.Labels]:
    return l if (l := _provide_active_layer()) and type(l) == layers.Labels else None  #noqa: E741


def _provide_active_points() -> Optional[layers.Points]:
    return l if (l := _provide_active_layer()) and type(l) == layers.Points else None  #noqa: E741


def _provide_active_shapes() -> Optional[layers.Shapes]:
    return l if (l := _provide_active_layer()) and type(l) == layers.Shapes else None  #noqa: E741


def _provide_active_surface() -> Optional[layers.Surface]:
    return l if (l := _provide_active_layer()) and type(l) == layers.Surface else None  #noqa: E741


def _provide_active_tracks() -> Optional[layers.Tracks]:
    return l if (l := _provide_active_layer()) and type(l) == layers.Tracks else None  #noqa: E741


def _provide_active_vectors() -> Optional[layers.Vectors]:
    return l if (l := _provide_active_layer()) and type(l) == layers.Vectors else None  #noqa: E741
# fmt: on


# syntax could be simplified after
# https://github.com/tlambert03/in-n-out/issues/31
PROVIDERS = [
    (_provide_viewer,),
    (_provide_active_layer,),
    (_provide_active_layer_list,),
    (_provide_active_image,),
    (_provide_active_labels,),
    (_provide_active_points,),
    (_provide_active_shapes,),
    (_provide_active_surface,),
    (_provide_active_tracks,),
    (_provide_active_vectors,),
]
