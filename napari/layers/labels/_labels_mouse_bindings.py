from concurrent.futures import ThreadPoolExecutor

from ._labels_constants import Mode
from ._labels_utils import interpolate_coordinates

refresh_executor = ThreadPoolExecutor(max_workers=1)


def refresh(layer):
    """Refresh the layer.

    Move layer refresh to a method executed on another thread, to relieve main
    event loop from heavy lifting and lagging due to cpu bound update through
    vispy, see https://github.com/napari/product-heuristics-2020/issues/38.
    (a better but not yet merged fix:https://github.com/vispy/vispy/pull/1912)

    """
    layer.refresh()


def draw(layer, event):
    """Draw with the currently selected label to a coordinate.

    This method have different behavior when draw is called
    with different labeling layer mode.

    In PAINT mode the cursor functions like a paint brush changing any
    pixels it brushes over to the current label. If the background label
    `0` is selected than any pixels will be changed to background and this
    tool functions like an eraser. The size and shape of the cursor can be
    adjusted in the properties widget.

    In FILL mode the cursor functions like a fill bucket replacing pixels
    of the label clicked on with the current label. It can either replace
    all pixels of that label or just those that are contiguous with the
    clicked on pixel. If the background label `0` is selected than any
    pixels will be changed to background and this tool functions like an
    eraser
    """
    # on press
    refresh_future = None
    layer._save_history()
    layer._block_saving = True
    if layer._mode == Mode.ERASE:
        new_label = layer._background_label
    else:
        new_label = layer.selected_label

    if layer._mode in [Mode.PAINT, Mode.ERASE]:
        layer.paint(layer.coordinates, new_label)
    elif layer._mode == Mode.FILL:
        layer.fill(layer.coordinates, new_label)

    last_cursor_coord = layer.coordinates
    yield

    # on move
    while event.type == 'mouse_move':
        interp_coord = interpolate_coordinates(
            last_cursor_coord, layer.coordinates, layer.brush_size
        )
        for c in interp_coord:
            if layer._mode in [Mode.PAINT, Mode.ERASE]:
                layer.paint(c, new_label, refresh=False)
            elif layer._mode == Mode.FILL:
                layer.fill(c, new_label, refresh=False)
        if not refresh_future or refresh_future.done():
            refresh_future = refresh_executor.submit(refresh, layer)
        last_cursor_coord = layer.coordinates
        yield

    # on release
    refresh_executor.submit(refresh, layer)
    layer._block_saving = False


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = layer._value or 0
