"""Debug overlay for progressive loading: chunk wireframes + level HUD.

Enable with ``NAPARI_PROGRESSIVE_DEBUG=1``, with
``add_progressive_loading_image(..., debug_overlay=True)``, or at
runtime via ``layer.metadata['progressive_loader'].enable_debug_overlay()``.

For the currently rendered level's resident interval this shows:

- one rectangle per chunk (a Shapes layer named ``"<layer> chunks"``),
  edge-colored by the chunk's content state:

  - green — real data at the rendered level
  - yellow — upsampled backdrop from one level coarser
  - orange — upsampled backdrop from two or more levels coarser
  - grey — content of unknown origin (e.g. carried over before
    provenance tracking saw it)
  - magenta — nothing recorded: this chunk would render as zeros

- the viewer text overlay reports the rendered vs. target level and the
  fetch ladder's stage progress.

Wireframes are drawn for 2D views; in 3D only the HUD text is shown.
The overlay polls a few times per second and only rebuilds shapes when
the underlying chunk bookkeeping actually changed.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QTimer

from napari.experimental._virtual_data import chunk_ids_in_region

if TYPE_CHECKING:
    from napari.experimental._progressive_loading import ProgressiveLoader

LOGGER = logging.getLogger(__name__)

#: RGBA edge colors per chunk content state.
STATE_COLORS = {
    'real': (0.15, 0.9, 0.25, 1.0),  # green: real data at this level
    'near': (1.0, 0.85, 0.1, 1.0),  # yellow: backdrop, 1 level coarser
    'far': (1.0, 0.5, 0.1, 1.0),  # orange: backdrop, >= 2 levels coarser
    'unknown': (0.6, 0.6, 0.6, 1.0),  # grey: content of unknown origin
    'empty': (1.0, 0.1, 0.9, 1.0),  # magenta: unfilled, renders as zeros
}
#: Above this many chunks the wireframes are skipped (HUD text remains):
#: rebuilding tens of thousands of shapes would itself cause stalls.
MAX_RECTS = 4096
POLL_MS = 250


class ChunkDebugOverlay:
    """Chunk wireframes plus a resolution HUD for a progressive layer.

    Normally created through
    :meth:`ProgressiveLoader.enable_debug_overlay`.
    """

    def __init__(self, loader: ProgressiveLoader):
        self._loader = loader
        self._viewer = loader._viewer
        self._layer = loader._layer
        self._shapes = None
        self._signature = None
        self._too_dense = False
        overlay = self._viewer.text_overlay
        self._saved_text_overlay = (overlay.visible, overlay.text)
        overlay.visible = True
        self._timer = QTimer()
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def close(self) -> None:
        """Stop polling, remove the shapes layer, restore the HUD."""
        self._timer.stop()
        with_timer, self._timer = self._timer, None
        with contextlib.suppress(RuntimeError, TypeError):
            with_timer.timeout.disconnect(self._refresh)
        viewer = self._viewer
        self._viewer = None
        shapes, self._shapes = self._shapes, None
        if viewer is None:
            return
        try:
            if shapes is not None and shapes in viewer.layers:
                viewer.layers.remove(shapes)
            visible, text = self._saved_text_overlay
            viewer.text_overlay.visible = visible
            viewer.text_overlay.text = text
        except Exception:  # noqa: BLE001 - viewer mid-teardown
            LOGGER.debug('debug overlay teardown incomplete', exc_info=True)

    # -- polling --

    def _refresh(self) -> None:
        loader = self._loader
        if self._viewer is None:
            return
        if loader._closed:
            self.close()
            return
        try:
            level = int(self._layer.data_level)
            vdata = loader._data[level]
            self._viewer.text_overlay.text = self._hud_text(level, vdata)
            signature = (
                level,
                vdata.interval,
                len(vdata.loaded_chunks),
                len(vdata.chunk_source),
                loader._chunks_done,
                self._viewer.dims.ndisplay,
            )
            if signature == self._signature:
                return
            self._signature = signature
            self._rebuild(level, vdata)
        except Exception:  # noqa: BLE001 - debug aid must never crash the app
            LOGGER.debug('debug overlay refresh failed', exc_info=True)

    def _hud_text(self, level: int, vdata) -> str:
        loader = self._loader
        target = loader._active[0] if loader._active is not None else level
        lines = [f'progressive: rendering L{level}, target L{target}']
        stages = loader._stages
        if loader._worker is not None and stages:
            index = min(loader._stage_index, len(stages) - 1)
            lines.append(
                f'pass {loader._chunks_done}/{loader._chunks_total} chunks '
                f'(stage {index + 1}/{len(stages)}: fetching '
                f'L{stages[index][0]})',
            )
        counts = ' '.join(
            f'L{lvl}:{len(loader._data[lvl].loaded_chunks)}'
            for lvl in range(len(loader._data))
        )
        lines.append(f'loaded chunks {counts}')
        if self._too_dense:
            lines.append(f'chunk grid > {MAX_RECTS}: wireframes hidden')
        return '\n'.join(lines)

    # -- wireframes --

    def _hide_shapes(self) -> None:
        if self._shapes is not None:
            self._shapes.visible = False

    def _chunk_state(self, vdata, level: int, chunk_id) -> str:
        if chunk_id in vdata.loaded_chunks:
            return 'real'
        source = vdata.chunk_source.get(chunk_id)
        if source is None:
            return 'empty'
        distance = source - level
        if distance <= 0:
            return 'real'  # carried-over content from this very level
        return 'near' if distance == 1 else 'far'

    def _rebuild(self, level: int, vdata) -> None:
        viewer = self._viewer
        if viewer.dims.ndisplay != 2 or vdata.interval is None:
            self._hide_shapes()
            return
        plane = self._loader._displayed_plane(level)
        if plane is None or len(plane[0]) != 2:
            self._hide_shapes()
            return
        displayed, steps = plane
        lo, hi = vdata.interval
        chunk_ids = [
            chunk_id
            for chunk_id in chunk_ids_in_region(vdata._boundaries, lo, hi)
            # only chunks on the rendered plane of non-displayed dims
            if all(
                chunk_id[d][0] <= step < chunk_id[d][1]
                for d, step in steps.items()
            )
        ]
        self._too_dense = len(chunk_ids) > MAX_RECTS
        if self._too_dense or not chunk_ids:
            self._hide_shapes()
            return

        layer = self._layer
        factors = np.asarray(
            self._loader._data._scale_factors[level],
            dtype=float,
        )
        scale = np.asarray(layer.scale, dtype=float)
        translate = np.asarray(layer.translate, dtype=float)
        d0, d1 = displayed

        def world(coord, d):
            return float(coord) * factors[d] * scale[d] + translate[d]

        rectangles = []
        colors = []
        for chunk_id in chunk_ids:
            (a0, b0), (a1, b1) = chunk_id[d0], chunk_id[d1]
            rectangles.append(
                [
                    [world(a0, d0), world(a1, d1)],
                    [world(a0, d0), world(b1, d1)],
                    [world(b0, d0), world(b1, d1)],
                    [world(b0, d0), world(a1, d1)],
                ],
            )
            colors.append(
                STATE_COLORS[self._chunk_state(vdata, level, chunk_id)]
            )

        edge_width = max(
            float(np.min([r[2][0] - r[0][0] for r in rectangles])) * 0.03,
            1e-6,
        )
        if self._shapes is None or self._shapes not in viewer.layers:
            active = viewer.layers.selection.active
            self._shapes = viewer.add_shapes(
                rectangles,
                shape_type='polygon',
                edge_color=colors,
                face_color=[0.0, 0.0, 0.0, 0.0],
                edge_width=edge_width,
                name=f'{layer.name} chunks',
                opacity=1.0,
            )
            self._shapes.editable = False
            # keep keyboard/mouse focus on the image layer under test
            if active is not None:
                viewer.layers.selection.active = active
        else:
            shapes = self._shapes
            shapes.visible = True
            shapes.data = rectangles
            shapes.edge_color = colors
            shapes.edge_width = edge_width
