import napari.utils.notifications
from napari._qt.qt_event_loop import gui_qt as gui_qt
from napari._qt.qt_event_loop import run as run
from napari.plugins.io import save_layers as save_layers
from napari.view_layers import (
    view_image as view_image,
)
from napari.view_layers import (
    view_labels as view_labels,
)
from napari.view_layers import (
    view_path as view_path,
)
from napari.view_layers import (
    view_points as view_points,
)
from napari.view_layers import (
    view_shapes as view_shapes,
)
from napari.view_layers import (
    view_surface as view_surface,
)
from napari.view_layers import (
    view_tracks as view_tracks,
)
from napari.view_layers import (
    view_vectors as view_vectors,
)
from napari.viewer import Viewer as Viewer
from napari.viewer import current_viewer as current_viewer

__version__: str

notification_manager: napari.utils.notifications.NotificationManager
