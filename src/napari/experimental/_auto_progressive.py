"""Route new multiscale layers through progressive loading.

When the ``experimental.progressive_loading`` setting is enabled
(Preferences > Experimental, or ``NAPARI_PROGRESSIVE_LOADING=1``),
chunked multiscale ``Image``/``Labels`` layers added to a viewer — by
drag-and-drop, ``File > Open``, plugin readers, or ``viewer.add_*`` —
are replaced with progressively loading equivalents built by
:func:`~napari.experimental._progressive_loading.add_progressive_loading_image`
(or ``..._labels``).

This module is deliberately import-light: the progressive loading
engine is only imported when the first layer is actually replaced, so
hooking a viewer costs nothing while the setting is off. The setting
is read per insertion, so toggling it affects layers added afterwards;
already-attached loaders keep running until their layer is removed.
"""

from __future__ import annotations

import logging
import weakref
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from napari.layers import Layer
    from napari.utils.events import Event

_logger = logging.getLogger(__name__)

# Guards against re-entry while add_progressive_loading_* inserts the
# replacement layer (which fires ``inserted`` again).  Per-viewer so
# concurrent viewers cannot bleed the guard into each other.
_attaching: set[int] = set()


def connect_viewer(viewer: napari.components.ViewerModel) -> None:
    """Watch *viewer* and progressively load new multiscale layers.

    A no-op per insertion unless the ``experimental.progressive_loading``
    setting is enabled.
    """
    viewer.layers.events.inserted.connect(
        partial(_on_inserted, weakref.ref(viewer))
    )


def _enabled() -> bool:
    from napari.settings import get_settings

    return bool(get_settings().experimental.progressive_loading)


def _eligible(layer: Layer) -> bool:
    """Whether *layer* should be replaced with a progressive one."""
    from napari.layers import Image, Labels

    if not isinstance(layer, Image | Labels) or not layer.multiscale:
        return False
    if 'progressive_loader' in layer.metadata:
        return False
    try:
        level0 = layer.data[0]
    except (IndexError, TypeError, KeyError):
        return False
    # Only chunked (lazy) pyramids benefit: zarr exposes ``chunks``,
    # dask ``chunksize``. Plain in-memory pyramids are left alone.
    return hasattr(level0, 'chunks') or hasattr(level0, 'chunksize')


def _on_inserted(viewer_ref: weakref.ref, event: Event) -> None:
    viewer = viewer_ref()
    if viewer is None:
        return
    if id(viewer) in _attaching or not _enabled():
        return
    layer = event.value
    if not _eligible(layer):
        return
    from qtpy.QtCore import QTimer

    # Defer the swap: the layer list must not be mutated while its own
    # ``inserted`` event is still being emitted.
    QTimer.singleShot(0, partial(_replace_with_progressive, viewer_ref, layer))


def _layer_kwargs(layer: Layer) -> dict:
    """Carry the original layer's appearance over to its replacement."""
    import numpy as np

    from napari.layers import Labels

    kwargs = {
        'name': layer.name,
        'opacity': layer.opacity,
        'blending': layer.blending,
        'visible': layer.visible,
        'metadata': dict(layer.metadata),
        'translate': tuple(layer.translate),
    }
    if not np.allclose(layer.scale, 1.0):
        # only pass a non-trivial scale: an explicit scale disables the
        # huge-world normalization in add_progressive_loading_image
        kwargs['scale'] = tuple(layer.scale)
    if not isinstance(layer, Labels):
        kwargs['contrast_limits'] = tuple(layer.contrast_limits)
        kwargs['colormap'] = layer.colormap.name
        kwargs['gamma'] = layer.gamma
        if layer.rgb:
            kwargs['rgb'] = True
        # ``rendering`` is intentionally not carried over: the
        # progressive default (attenuated_mip) is tuned for streaming.
    return kwargs


def _replace_with_progressive(viewer_ref: weakref.ref, layer: Layer) -> None:
    viewer = viewer_ref()
    if viewer is None or layer not in viewer.layers:
        return
    # re-check: this runs deferred, state may have changed
    if not _enabled() or not _eligible(layer):
        return
    from napari.experimental._progressive_loading import (
        add_progressive_loading_image,
        add_progressive_loading_labels,
    )
    from napari.layers import Labels

    data = list(layer.data)
    kwargs = _layer_kwargs(layer)
    index = viewer.layers.index(layer)
    vid = id(viewer)
    _attaching.add(vid)
    try:
        viewer.layers.remove(layer)
        try:
            if isinstance(layer, Labels):
                add_progressive_loading_labels(data, viewer=viewer, **kwargs)
            else:
                add_progressive_loading_image(data, viewer=viewer, **kwargs)
        except Exception:
            _logger.exception(
                'progressive loading failed for layer %r; '
                'restoring the original layer',
                kwargs['name'],
            )
            viewer.layers.insert(index, layer)
            return
        # the replacement was appended at the end; restore the position
        viewer.layers.move(len(viewer.layers) - 1, index)
    finally:
        _attaching.discard(vid)
