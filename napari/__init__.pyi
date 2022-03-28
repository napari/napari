import napari._qt.qt_event_loop
import napari.plugins.io
import napari.utils.notifications
import napari.view_layers
import napari.viewer

__version__: str

notification_manager: napari.utils.notifications.NotificationManager
Viewer = napari.viewer.Viewer
current_viewer = napari.viewer.current_viewer

gui_qt = napari._qt.qt_event_loop.gui_qt
run = napari._qt.qt_event_loop.run
save_layers = napari.plugins.io.save_layers

view_image = napari.view_layers.view_image
view_labels = napari.view_layers.view_labels
view_path = napari.view_layers.view_path
view_points = napari.view_layers.view_points
view_shapes = napari.view_layers.view_shapes
view_surface = napari.view_layers.view_surface
view_tracks = napari.view_layers.view_tracks
view_vectors = napari.view_layers.view_vectors
