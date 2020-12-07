"""QtOctreeInfo and QtOctreeInfoLayout classes.

Shows octree-specific information in the QtRender widget.
"""
import numpy as np
from qtpy.QtWidgets import QCheckBox, QFrame, QVBoxLayout

from ....components.experimental import chunk_loader
from ....layers.image.experimental.octree_image import OctreeImage
from .qt_render_widgets import QtLabeledComboBox, QtSimpleTable


def _get_table_values(layer: OctreeImage) -> dict:
    """Get keys/values about this octree image for the table.

    layer : OctreeImage
        Get values for this octree image.
    """

    def _shape(shape) -> str:
        return f"{shape[1]}x{shape[0]}"

    level_info = layer.octree_level_info

    if level_info is None:
        return {}

    shape_in_tiles = level_info.shape_in_tiles
    num_tiles = shape_in_tiles[0] * shape_in_tiles[1]

    return {
        "Level": f"{layer.data_level}",
        "Tiles": f"{_shape(shape_in_tiles)} = {num_tiles}",
        "Tile Shape": _shape([layer.tile_size, layer.tile_size]),
        "Layer Shape": _shape(level_info.image_shape),
    }


class QtOctreeInfoLayout(QVBoxLayout):
    """OctreeImage specific information.

    Combo base to choose octree layer or set to AUTO for the normal rendering
    mode where the correct level is chosen automatically. (not working yet)

    Parameters
    ----------
    layer : OctreeImage
        Show octree info for this layer.
    on_set_level : Callable[[int], None]
        Call this when the octree level is changed.
    """

    def __init__(self, layer: OctreeImage):
        super().__init__()

        self.layer = layer

        def on_set_cache(value: int) -> None:
            chunk_loader.cache.enabled = value != 0

        def on_set_grid(value: int) -> None:
            self.layer.display.show_grid = value != 0

        def on_set_level(value: int) -> None:
            if value == 0:  # This is AUTO
                self.layer.display.freeze_level = False
            else:
                level = value - 1  # Account for AUTO at 0
                self.layer.display.freeze_level = True
                self.layer.octree_level = level

        def on_set_track(value: int):
            self.layer.display.track_view = value != 0

        # Toggle the ChunkCache.
        cache_enabled = chunk_loader.cache.enabled
        self._create_checkbox("Chunk Cache", cache_enabled, on_set_cache)

        # Toggle tracking: drawn tiles track view as it moves.
        self._create_checkbox(
            "Track View", layer.display.track_view, on_set_track
        )

        # Toggle debug grid drawn around tiles.
        self._create_checkbox(
            "Show Grid", layer.display.show_grid, on_set_grid
        )

        num_levels = layer.num_octree_levels

        # Show AUTO followed by the valid layer numbers we can choose. AUTO
        # means OctreeImage selects the appropriate octree level
        # dynamically as you zoom in or out. Which is the normal behavior
        # the user almost always wants.
        level_options = {"AUTO": -1}
        level_options.update({str(x): x for x in np.arange(0, num_levels)})
        self.level = QtLabeledComboBox(
            "Octree Level", level_options, on_set_level
        )
        self.addWidget(self.level)

        # Show some keys and values about the octree.
        self.table = QtSimpleTable()
        self.addWidget(self.table)

        self.set_controls(layer)  # Initial settings.

    def _create_checkbox(self, label, initial_value, callback):
        checkbox = QCheckBox(label)
        checkbox.stateChanged.connect(callback)
        checkbox.setChecked(initial_value)
        self.addWidget(checkbox)

    def set_controls(self, layer: OctreeImage):
        """Set controls based on the layer.

        Parameters
        ----------
        layer : OctreeImage
            Set controls based on this layer.
        """
        self.level.set_value(
            "AUTO" if not layer.display.freeze_level else layer.octree_level
        )
        self.table.set_values(_get_table_values(layer))


class QtOctreeInfo(QFrame):
    """Frame showing the octree level and tile size.

    layer : OctreeImage
        Show info about this layer.
    """

    def __init__(self, layer: OctreeImage):
        super().__init__()
        self.layer = layer

        self.layout = QtOctreeInfoLayout(layer)
        self.setLayout(self.layout)

        # Initial update and connect for future updates.
        self._update()  # initial update

        # removing this event...
        # layer.events.freeze_level.connect(self._update)

        layer.events.octree_level.connect(self._update)
        layer.events.tile_size.connect(self._update)

    def _update(self, _event=None):
        """Set controls based on the current layer setting."""
        self.layout.set_controls(self.layer)
