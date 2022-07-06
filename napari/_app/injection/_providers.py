from functools import lru_cache
from typing import Optional

from in_n_out import Store

from ... import components, layers, viewer


def _provide_viewer() -> Optional[viewer.Viewer]:
    return Store.get_store('napari').provide(viewer.Viewer)


def _provide_active_layer() -> Optional[layers.Layer]:
    return v.layers.selection.active if (v := _provide_viewer()) else None


def _provide_active_layer_list() -> Optional[components.LayerList]:
    return v.layers if (v := _provide_viewer()) else None


@lru_cache
def _init_providers(store: Store):
    from ... import viewer

    # TODO: fix syntax for this upstream
    store.register(
        providers=[
            (viewer.current_viewer,),
            (_provide_active_layer,),
            (_provide_active_layer_list,),
        ]
    )
