from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, PrivateAttr, field_validator

from napari.utils.camera_orientations import (
    DEFAULT_ORIENTATION_TYPED,
    DepthAxisOrientation,
    Handedness,
    HorizontalAxisOrientation,
    HorizontalAxisOrientationStr,
    VerticalAxisOrientation,
    VerticalAxisOrientationStr,
)
from napari.utils.events import EventedModel
from napari.utils.misc import ensure_n_tuple

if TYPE_CHECKING:
    import numpy.typing as npt


_SYNCED_CAMERA_DESCRIPTION = (
    'Controls how camera state is managed when switching between\n'
    '2D and 3D views. When checked, camera center and zoom are\n'
    'shared between views, with the depth (Z) component synced via\n'
    'the dims slider. When unchecked, each mode remembers\n'
    'its own camera state independently.'
)


@dataclass(frozen=True)
class _CameraState:
    """Captured camera state for a single ndisplay mode.

    This is a lightweight private data container used internally by
    :class:`Camera` to preserve per-mode center, zoom, and angles when
    switching between 2D and 3D views.
    """

    center: tuple[float, float, float] | tuple[float, float]
    zoom: float
    angles: tuple[float, float, float]


class Camera(EventedModel):
    """Camera object modeling position and view of the camera.

    Attributes
    ----------
    center : 3-tuple
        Center of rotation for the camera.
        In 2D viewing the last two values are used.
    zoom : float
        Scale from canvas pixels to world pixels.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.
        Euler angles in 3D do not uniquely represent an orientation, so
        different angle triplets can produce the same view.
        Stored or returned angle values may differ from those that were set,
        while still representing an equivalent camera orientation.
    perspective : float
        Perspective (aka "field of view" in vispy) of the camera (if 3D).
    mouse_pan : bool
        If the camera interactive panning with the mouse is enabled or not.
    mouse_zoom : bool
        If the camera interactive zooming with the mouse is enabled or not.
    """

    # fields
    center: tuple[float, float, float] | tuple[float, float] = (
        0.0,
        0.0,
        0.0,
    )
    zoom: float = 1.0
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    perspective: float = 0
    mouse_pan: bool = True
    mouse_zoom: bool = True
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ] = DEFAULT_ORIENTATION_TYPED
    synced: bool = Field(True, description=_SYNCED_CAMERA_DESCRIPTION)

    # Per-mode camera state cache for the "separate" (synced=False) mode.
    _cached_2d_state: _CameraState | None = PrivateAttr(None)
    _cached_3d_state: _CameraState | None = PrivateAttr(None)

    def _cache_state(self, ndisplay_mode: int) -> None:
        """Save current camera state for a given ndisplay mode."""
        state = _CameraState(
            center=self.center,
            zoom=self.zoom,
            angles=self.angles,
        )
        if ndisplay_mode == 2:
            self._cached_2d_state = state
        else:
            self._cached_3d_state = state

    def _pop_cached_state(self, ndisplay_mode: int) -> _CameraState | None:
        """Retrieve and remove cached state for a given ndisplay mode."""
        if ndisplay_mode == 2:
            state = self._cached_2d_state
            self._cached_2d_state = None
        else:
            state = self._cached_3d_state
            self._cached_3d_state = None
        return state

    @field_validator('center', 'angles', mode='before')
    @classmethod
    def _ensure_3_tuple(cls, v):
        return ensure_n_tuple(v, n=3)

    @property
    def view_direction(self) -> tuple[float, float, float]:
        """3D view direction vector of the camera.

        View direction is calculated from the Euler angles and returned as a
        3-tuple. This direction is in 3D scene coordinates, the world coordinate
        system for three currently displayed dimensions.
        """
        from scipy.spatial.transform import Rotation as R

        # once we're in scene-land, we pretend to be in xyz space (axes names don't
        # mean anything after all...) which simplifies the logic a lot.
        rotation = R.from_euler('xyz', self.angles, degrees=True)
        # view direction is given by the z component, but flipping the sign.
        # This is because the default view direction at angles (0, 0, 0) is (-1, 0, 0)
        return tuple(-rotation.as_matrix()[0])

    @property
    def up_direction(self) -> tuple[float, float, float]:
        """3D direction vector pointing up on the canvas.

        Up direction is calculated from the Euler angles and returned as a
        3-tuple. This direction is in 3D scene coordinates, the world coordinate
        system for three currently displayed dimensions.
        """
        from scipy.spatial.transform import Rotation as R

        # once we're in scene-land, we pretend to be in xyz space (axes names don't
        # mean anything after all...) which simplifies the logic a lot.
        rotation = R.from_euler('xyz', self.angles, degrees=True)
        # up direction is given by the y component, but flipping the sign.
        # This is because the default up direction at angles (0, 0, 0) is (0, -1, 0)
        return tuple(-rotation.as_matrix()[1])

    def set_view_direction(
        self,
        view_direction: tuple[float, float, float],
        up_direction: tuple[float, float, float] = (0, -1, 0),
    ):
        """Set camera angles from direction vectors.

        Both the view direction and the up direction are specified in 3D scene
        coordinates, the world coordinate system for three currently displayed
        dimensions.

        The provided up direction must not be parallel to the provided
        view direction. The provided up direction does not need to be orthogonal
        to the view direction. The final up direction will be a vector orthogonal
        to the view direction, aligned with the provided up direction.

        Parameters
        ----------
        view_direction : 3-tuple of float
            The desired view direction vector in 3D scene coordinates, the world
            coordinate system for three currently displayed dimensions.
        up_direction : 3-tuple of float
            A direction vector which will point upwards on the canvas. Defaults
            to (0, -1, 0) unless the view direction is parallel to the y-axis,
            in which case will default to (-1, 0, 0).
        """
        from scipy.spatial.transform import Rotation as R

        # project up onto view so we can remove the parallel component
        projection = np.dot(up_direction, view_direction) * np.array(
            view_direction
        )
        up_direction_arr = np.asarray(up_direction) - projection

        view_direction_arr = np.asarray(view_direction) / np.linalg.norm(
            view_direction
        )
        up_direction_arr = up_direction_arr / np.linalg.norm(up_direction_arr)
        right_direction = np.cross(up_direction_arr, view_direction_arr)

        # once we're in scene-land, we pretend to be in xyz space (axes names don't
        # mean anything after all...) which simplifies the logic a lot. We also
        # flip all signs (see explanations in self.view_direction, and self.up_direction)
        matrix = -np.array(
            (view_direction_arr, up_direction_arr, right_direction)
        )
        self.angles = R.from_matrix(matrix).as_euler('xyz', degrees=True)

    def calculate_nd_view_direction(
        self,
        ndim: int,
        dims_displayed: tuple[int, ...],
        canvas_position: tuple[float, float] | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> npt.NDArray[np.float64] | None:
        """Calculate the nD view direction vector of the camera.

        When ``canvas_position`` and ``canvas_size`` are given and the camera
        uses a perspective projection, the view direction is calculated for the
        ray going from the eye through that canvas position (accounting for the
        field of view). Otherwise, the view direction is that of the center of
        the view.

        Parameters
        ----------
        ndim : int
            Number of dimensions in which to embed the 3D view vector.
        dims_displayed : Tuple[int]
            Dimensions in which to embed the 3D view vector.
        canvas_position : tuple of float, optional
            Position in the canvas in pixels, as ``(x, y)`` where ``x`` is the
            column and ``y`` is the row. If ``None``, the view direction is
            calculated for the center of the canvas.
        canvas_size : tuple of int, optional
            Size of the canvas in pixels, as ``(height, width)``. Only used when
            ``canvas_position`` is given.

        Returns
        -------
        view_direction_nd : np.ndarray
            nD view direction vector as an (ndim, ) ndarray
        """
        if len(dims_displayed) != 3:
            return None
        view_direction_nd = np.zeros(ndim)
        if (
            canvas_position is not None
            and canvas_size is not None
            and self.perspective > 0
        ):
            view_direction = self._view_direction_at(
                canvas_position, canvas_size
            )
        else:
            view_direction = np.asarray(self.view_direction)
        view_direction_nd[list(dims_displayed)] = view_direction
        return view_direction_nd

    def _view_direction_at(
        self,
        canvas_position: tuple[float, float],
        canvas_size: tuple[int, int],
    ) -> npt.NDArray[np.float64]:
        """Calculate the view direction of the ray from the eye through a canvas position.

        Parameters
        ----------
        canvas_position : tuple of float
            Position in the canvas in pixels, as ``(x, y)`` where ``x`` is the
            column and ``y`` is the row.
        canvas_size : tuple of int
            Size of the canvas in pixels, as ``(height, width)``.

        Returns
        -------
        view_direction : np.ndarray
            Normalized 3D view direction vector in scene coordinates, in the
            world coordinate system of the three displayed dimensions.
        """
        x, y = canvas_position
        h, w = canvas_size

        view_direction = np.asarray(self.view_direction)
        up_direction = np.asarray(self.up_direction)
        # vector pointing to the right of the canvas, in scene coordinates
        right_direction = np.cross(view_direction, up_direction)

        # distance of the eye from the center of the view, in world
        # coordinates. This matches the perspective projection used by
        # vispy, combined with the zoom-to-scale factor relation
        # (see napari/_vispy/camera.py).
        dist = h / (2 * self.zoom * np.tan(np.radians(self.perspective) / 2))

        # offset of the canvas position from the canvas center, in world
        # coordinates
        dx = (x - w / 2) / self.zoom
        dy = (y - h / 2) / self.zoom

        # ray direction from the eye through the canvas position
        ray = dist * view_direction + dx * right_direction - dy * up_direction
        return ray / np.linalg.norm(ray)

    def calculate_nd_up_direction(
        self, ndim: int, dims_displayed: tuple[int, ...]
    ) -> np.ndarray | None:
        """Calculate the nD up direction vector of the camera.

        Parameters
        ----------
        ndim : int
            Number of dimensions in which to embed the 3D view vector.
        dims_displayed : Tuple[int]
            Dimensions in which to embed the 3D view vector.

        Returns
        -------
        up_direction_nd : np.ndarray
            nD view direction vector as an (ndim, ) ndarray
        """
        if len(dims_displayed) != 3:
            return None
        up_direction_nd = np.zeros(ndim)
        up_direction_nd[list(dims_displayed)] = self.up_direction
        return up_direction_nd

    @property
    def orientation2d(
        self,
    ) -> tuple[VerticalAxisOrientation, HorizontalAxisOrientation]:
        return self.orientation[1:]

    @orientation2d.setter
    def orientation2d(
        self,
        value: tuple[
            VerticalAxisOrientation | VerticalAxisOrientationStr,
            HorizontalAxisOrientation | HorizontalAxisOrientationStr,
        ],
    ) -> None:
        self.orientation = (
            self.orientation[0],
            VerticalAxisOrientation(value[0]),
            HorizontalAxisOrientation(value[1]),
        )

    @property
    def handedness(self) -> Handedness:
        """Right or left-handedness of the current orientation."""
        # we know default orientation is right-handed, so an odd number of
        # differences from default means left-handed.
        diffs = [
            self.orientation[i] != DEFAULT_ORIENTATION_TYPED[i]
            for i in range(3)
        ]
        if sum(diffs) % 2 != 0:
            return Handedness.LEFT
        return Handedness.RIGHT
