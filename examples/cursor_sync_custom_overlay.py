import warnings

import numpy as np
from vispy.scene.visuals import Ellipse

import napari
from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.utils.visual import overlay_to_visual
from napari.components.overlays import CanvasOverlay
from napari.utils.events import Event

#
#   Create a cursor overlay
#


class CursorOverlay(CanvasOverlay):
    """Used as the key for overlay_to_visual's overlay dict"""


class VispyCursorOverlay(ViewerOverlayMixin, VispyCanvasOverlay):

    def __init__(self, *, viewer, overlay, parent=None):

        cursor = Ellipse(center=(4, 4), color="red", radius=4)

        super().__init__(
            node=cursor,
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self.reset()

    def reset(self):
        super().reset()


#
#   Register the new "cursor" in the global `overlay_to_visual`
#

overlay_to_visual[CursorOverlay] = VispyCursorOverlay  # type: ignore


#
#   Wire it up
#

def sync_cursor(parent: napari.Viewer, child: napari.Viewer):
    """Syncs a 'cursor' onto the child, relative to the parent's canvas"""

    cursor_overlay = CursorOverlay(visible=True)  # create a new "cursor" overlay
    child._overlays['t_cursor'] = cursor_overlay  # add to the child
    child.window._qt_viewer._add_overlay(cursor_overlay)  # trigger overlay generation

    @parent.cursor.events.position.connect
    def sync_cursor_parent_callback(e: Event):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            t_cursor: CursorOverlay = child._overlays['t_cursor']
            cursor: VispyCursorOverlay = child.window._qt_viewer.overlay_to_visual[t_cursor]

            # Get position of real cursor relative to the parent's canvas
            parent_global_pos = parent.window._qt_viewer.cursor().pos()
            parent_local_pos = parent.window._qt_viewer.mapFromGlobal(parent_global_pos)

            child_x = parent_local_pos.x()
            child_y = parent_local_pos.y()

            # Apply new position to "cursor"
            transform = [child_x, child_y, 0, 0]
            cursor.node.transform.translate = transform


#
#   Run it
#

if __name__ == "__main__":

    parent = napari.Viewer(title="Parent")
    child = napari.Viewer(title="Child")

    parent_layer = parent.add_labels(np.random.randint(0, 10, (500, 300)))
    child_layer = child.add_labels(np.random.randint(10, 20, (500, 300)))

    sync_cursor(parent, child)

    napari.run()
