from typing import Optional

from ... import components, layers, viewer


def _provide_viewer() -> Optional[viewer.Viewer]:
    return viewer.current_viewer()


def _provide_active_layer() -> Optional[layers.Layer]:
    return v.layers.selection.active if (v := _provide_viewer()) else None


def _provide_active_layer_list() -> Optional[components.LayerList]:
    return v.layers if (v := _provide_viewer()) else None


# TODO: fix syntax for this upstream
PROVIDERS = [
    (_provide_viewer,),
    (_provide_active_layer,),
    (_provide_active_layer_list,),
]
