from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from vispy.scene import ArcballCamera, BaseCamera, PanZoomCamera
from vispy.util.quaternion import Quaternion

from napari.utils.camera_orientations import (
    angles_from_view_direction,
    up_direction_from_angles,
    view_direction_from_angles,
)

if TYPE_CHECKING:
    from napari.utils.camera_orientations import (
        DepthAxisOrientation,
        HorizontalAxisOrientation,
        VerticalAxisOrientation,
    )


def _get_vispy_flipped_axes(
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
    ndisplay: Literal[2, 3] = 2,
) -> tuple[int, int, int]:
    # Note that the default Vispy camera orientations come from Vispy, not from us.
    vispy_default_orientation = (
        ('right', 'up', 'towards')
        if ndisplay == 2
        else ('right', 'down', 'away')
    )

    # Vispy uses xyz coordinates; napari uses zyx coordinates, so we invert
    orientation_xyz = orientation[::-1]
    # The Vispy camera flip is a tuple of three ints in {0, 1}, indicating
    # whether they are flipped relative to the Vispy default.
    return cast(
        tuple[int, int, int],
        tuple(
            int(ori != default_ori)
            for ori, default_ori in zip(
                orientation_xyz, vispy_default_orientation, strict=True
            )
        ),
    )


def _quaternion_to_matrix(
    quat: Quaternion,
) -> np.ndarray:
    """Return the rotation matrix for a VisPy quaternion.

    VisPy derives its rotation from the quaternion axis-angle, permuting the
    vector part as (x, z, y) [NOTE THE WRONG ORDER].
    See #8281 for the origin of this quirk.

    Parameters
    ----------
    quat : vispy.util.quaternion.Quaternion
        The VisPy quaternion as rendered by the VisPy 3D camera.

    Returns
    -------
    np.ndarray
        The (3, 3) rotation matrix in VisPy (xyz) coordinates.
    """
    from scipy.spatial.transform import Rotation

    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    return Rotation.from_quat([x, z, y, w]).as_matrix().T


def _matrix_to_quaternion(
    matrix: np.ndarray,
) -> Quaternion:
    """Return the VisPy quaternion for the given rotation matrix.

    The inverse of :func:`_quaternion_to_matrix`.
    """
    from scipy.spatial.transform import Rotation

    qx, qy, qz, qw = Rotation.from_matrix(matrix.T).as_quat()
    # VisPy uses the vector part permuted as (x, z, y).
    return Quaternion(qw, qx, qz, qy)


def _directions_to_vispy_quat(
    view_direction: tuple[float, float, float],
    up_direction: tuple[float, float, float],
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> Quaternion:
    """Return the VisPy quaternion from the given napari view and up directions."""
    flipped_axes = _get_vispy_flipped_axes(orientation, ndisplay=3)
    factors = np.where(flipped_axes, -1, 1)
    # zyx -> xyz flip
    view_row = factors * np.asarray(view_direction)[::-1]
    up_row = factors * np.asarray(up_direction)[::-1]
    matrix = np.stack([np.cross(view_row, up_row), view_row, up_row])
    return _matrix_to_quaternion(matrix)


def napari_angles_to_vispy_quat(
    angles: tuple[float, float, float],
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> Quaternion:
    """Return the VisPy quaternion for the given napari camera angles.

    Parameters
    ----------
    angles : 3-tuple of float
        Euler angles of the 3D camera, in degrees.
    orientation : 3-tuple of str
        The napari orientation, with depth, vertical, and horizontal components,
        in napari (zyx) order.

    Returns
    -------
    vispy.util.quaternion.Quaternion
        The VisPy quaternion rendering the given camera angles.
    """
    return _directions_to_vispy_quat(
        view_direction_from_angles(angles, orientation),
        up_direction_from_angles(angles, orientation),
        orientation,
    )


def vispy_quat_to_napari_angles(
    quat: Quaternion,
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> tuple[float, float, float]:
    """Return the napari camera angles for the given VisPy quaternion.

    Parameters
    ----------
    quat : vispy.util.quaternion.Quaternion
        The VisPy quaternion as rendered by the VisPy 3D camera.
    orientation : 3-tuple of str
        The napari orientation, with depth, vertical, and horizontal components,
        in napari (zyx) order.

    Returns
    -------
    3-tuple of float
        Euler angles (rx, ry, rz) of the 3D camera, in degrees.
    """
    matrix = _quaternion_to_matrix(quat)
    flipped_axes = _get_vispy_flipped_axes(orientation, ndisplay=3)
    factors = np.where(flipped_axes, -1, 1)
    # Undo the flips and switch from VisPy (xyz) to napari (zyx) coordinates.
    view_direction = tuple((factors * matrix[1])[::-1])
    up_direction = tuple((factors * matrix[2])[::-1])
    return angles_from_view_direction(
        view_direction, up_direction, orientation
    )


class VispyCamera:
    """Vipsy camera for both 2D and 3D rendering.

    Parameters
    ----------
    view : vispy.scene.widgets.viewbox.ViewBox
        Viewbox for current scene.
    camera : napari.components.Camera
        napari camera model.
    dims : napari.components.Dims
        napari dims model.
    """

    def __init__(self, view, camera, dims) -> None:
        self._view = view
        self._camera = camera
        self._dims = dims

        # Create 2D camera
        self._2D_camera = MouseToggledPanZoomCamera(aspect=1)

        # Create 3D camera
        self._3D_camera = MouseToggledArcballCamera(fov=0)

        self._view.camera = (
            self._2D_camera if dims.ndisplay == 2 else self._3D_camera
        )
        self._last_viewbox_size = np.array((0, 0))

        self._dims.events.ndisplay.connect(
            self._on_ndisplay_change, position='first'
        )
        self._camera.events.center.connect(self._on_center_change)
        self._camera.events.zoom.connect(self._on_zoom_change)
        self._camera.events.angles.connect(self._on_angles_change)
        self._camera.events.perspective.connect(self._on_perspective_change)
        self._camera.events.mouse_pan.connect(self._on_mouse_toggles_change)
        self._camera.events.mouse_zoom.connect(self._on_mouse_toggles_change)
        self._camera.events.orientation.connect(self._on_orientation_change)

    @property
    def angles(self):
        """3-tuple: Euler angles of camera in 3D viewing, in degrees.
        Note that angles might be different than the ones that might have generated the quaternion.
        """

        if isinstance(self._view.camera, MouseToggledArcballCamera):
            return vispy_quat_to_napari_angles(
                self._view.camera._quaternion, self._camera.orientation
            )

        return (0, 0, 0)

    @angles.setter
    def angles(self, angles):
        if self.angles == tuple(angles):
            return

        # Only update angles if current camera is 3D camera
        if isinstance(self._view.camera, MouseToggledArcballCamera):
            quat = napari_angles_to_vispy_quat(
                self._camera.angles, self._camera.orientation
            )
            self._view.camera.set_state(_quaternion=quat)

    @property
    def center(self):
        """tuple: Center point of camera view for 2D or 3D viewing."""
        if isinstance(self._view.camera, MouseToggledArcballCamera):
            center = tuple(self._view.camera.center)
        else:
            # in 2D, we arbitrarily choose 0.0 as the center in z
            center = (*self._view.camera.center[:2], 0.0)
        # switch from VisPy xyz ordering to NumPy prc ordering
        return center[::-1]

    @center.setter
    def center(self, center):
        if self.center == tuple(center):
            return
        self._view.camera.center = center[::-1]
        self._view.camera.view_changed()

    @property
    def zoom(self):
        """float: Scale from canvas pixels to world pixels."""
        viewbox_size = np.array(self._view.rect.size)
        if isinstance(self._view.camera, MouseToggledArcballCamera):
            # For fov = 0.0 normalize scale factor by canvas size to get scale factor.
            # Note that the scaling is stored in the `_projection` property of the
            # camera which is updated in vispy here
            # https://github.com/vispy/vispy/blob/v0.6.5/vispy/scene/cameras/perspective.py#L301-L313
            scale = self._view.camera.scale_factor
        else:
            scale = np.array(
                [self._view.camera.rect.width, self._view.camera.rect.height]
            )
            # we just never want this to be zero; however, ideally we should block zooming
            # itself (currently we rely on vispy to do camera callbacks so we can't do this)
            # other stuff breaks at lower than 1e-8, so we shhouldn't go there...
            scale = np.clip(scale, 1e-8, np.inf)
        zoom = np.min(viewbox_size / scale)
        return zoom

    @zoom.setter
    def zoom(self, zoom):
        if self.zoom == zoom:
            return
        viewbox_size = np.array(self._view.rect.size)
        scale = np.array(viewbox_size) / zoom
        if isinstance(self._view.camera, MouseToggledArcballCamera):
            self._view.camera.scale_factor = np.min(scale)
        else:
            # Set view rectangle, as left, right, width, height
            corner = np.subtract(self._view.camera.center[:2], scale / 2)
            self._view.camera.rect = tuple(corner) + tuple(scale)

    @property
    def perspective(self):
        """Field of view of camera (only visible in 3D mode)."""
        return self._3D_camera.fov

    @perspective.setter
    def perspective(self, perspective):
        if self.perspective == perspective:
            return
        self._3D_camera.fov = perspective
        self._view.camera.view_changed()

    @property
    def mouse_zoom(self) -> bool:
        return self._view.camera.mouse_zoom

    @mouse_zoom.setter
    def mouse_zoom(self, mouse_zoom: bool):
        self._view.camera.mouse_zoom = mouse_zoom

    @property
    def mouse_pan(self) -> bool:
        return self._view.camera.mouse_pan

    @mouse_pan.setter
    def mouse_pan(self, mouse_pan: bool):
        self._view.camera.mouse_pan = mouse_pan

    def _on_ndisplay_change(self):
        # remove previous camera from children
        self._view.camera.parent = None
        if self._dims.ndisplay == 3:
            self._view.camera = self._3D_camera
            self._on_angles_change()
        else:
            self._view.camera = self._2D_camera

        self._on_mouse_toggles_change()
        self._on_center_change()
        self._on_zoom_change()
        self._on_orientation_change()
        self._on_perspective_change()

    def _on_mouse_toggles_change(self):
        self.mouse_pan = self._camera.mouse_pan
        self.mouse_zoom = self._camera.mouse_zoom

    def _on_center_change(self):
        self.center = self._camera.center[-self._dims.ndisplay :]

    def _on_zoom_change(self):
        self.zoom = self._camera.zoom

    def _on_orientation_change(self):
        self._2D_camera.flip = _get_vispy_flipped_axes(
            self._camera.orientation, ndisplay=2
        )
        self._3D_camera.flip = _get_vispy_flipped_axes(
            self._camera.orientation, ndisplay=3
        )

    def _on_perspective_change(self):
        self.perspective = self._camera.perspective

    def _on_angles_change(self):
        with self._camera.events.angles.blocker():
            self.angles = self._camera.angles

    def on_draw(self, _event):
        """Called whenever the canvas is drawn.

        Update camera model angles, center, and zoom.
        """
        # if the viewboxsize changed since last time, we need to update
        viewbox_size = np.array(self._view.rect.size)
        if not np.allclose(self._last_viewbox_size, viewbox_size):
            self._last_viewbox_size = viewbox_size
            self._on_ndisplay_change()

        if not np.allclose(self.angles, self._camera.angles) and isinstance(
            self._view.camera,
            MouseToggledArcballCamera,
        ):
            with self._camera.events.angles.blocker(self._on_angles_change):
                self._camera.angles = self.angles
        if not np.allclose(self.center, self._camera.center):
            with self._camera.events.center.blocker(self._on_center_change):
                self._camera.center = self.center
        if not np.allclose(self.zoom, self._camera.zoom):
            with self._camera.events.zoom.blocker(self._on_zoom_change):
                self._camera.zoom = self.zoom
        if not np.allclose(self.perspective, self._camera.perspective):
            with self._camera.events.perspective.blocker(
                self._on_perspective_change
            ):
                self._camera.perspective = self.perspective


def add_mouse_pan_zoom_toggles(
    vispy_camera_cls: type[BaseCamera],
) -> type[BaseCamera]:
    """Add separate mouse pan and mouse zoom toggles to VisPy.

    By default, VisPy uses an ``interactive`` toggle that turns *both*
    panning and zooming on and off. This decorator adds separate toggles,
    ``mouse_pan`` and ``mouse_zoom``, to enable controlling them
    separately.

    This also overrides viewbox_mouse_event and viewbox_key_event which are
    called unnecessarily for us and cause exceptions in some cases.

    Parameters
    ----------
    vispy_camera_cls : Type[vispy.scene.cameras.BaseCamera]
        A VisPy camera class to decorate.

    Returns
    -------
    A decorated VisPy camera class.
    """

    class _vispy_camera_cls(vispy_camera_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mouse_pan = True
            self.mouse_zoom = True

        def viewbox_mouse_event(self, event):
            if (
                self.mouse_zoom
                and event.type in ('mouse_wheel', 'gesture_zoom')
            ) or (
                self.mouse_pan
                and event.type
                in ('mouse_move', 'mouse_press', 'mouse_release')
            ):
                super().viewbox_mouse_event(event)
            else:
                event.handled = False

        def viewbox_resize_event(self, event):
            # due to the 2d/3d switching of cameras, sometimes we momentarily
            # try to update cameras that are not in the scenegraph;
            # in that case, we can just skip the update
            if self not in event.source.scene.children:
                return
            super().viewbox_resize_event(event)

        def viewbox_key_event(self, event):
            """ViewBox key event handler.

            Parameters
            ----------
            event : vispy.util.event.Event
                The vispy event that triggered this method.
            """
            return

    return _vispy_camera_cls


MouseToggledPanZoomCamera = add_mouse_pan_zoom_toggles(PanZoomCamera)
MouseToggledArcballCamera = add_mouse_pan_zoom_toggles(ArcballCamera)
