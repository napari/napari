from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from scipy.spatial.transform import Rotation as R

from napari._pydantic_compat import validator
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
        Note that Euler angles's intrinsic degeneracy means different
        sets of Euler angles may lead to the same view.
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

    # validators
    @validator('center', 'angles', pre=True, allow_reuse=True)
    def _ensure_3_tuple(cls, v):
        return ensure_n_tuple(v, n=3)

    @property
    def view_direction(self) -> tuple[float, float, float]:
        """3D view direction vector of the camera.

        View direction is calculated from the Euler angles and returned as a
        3-tuple. This direction is in 3D scene coordinates, the world coordinate
        system for three currently displayed dimensions.
        """
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
        self, ndim: int, dims_displayed: tuple[int, ...]
    ) -> npt.NDArray[np.float64] | None:
        """Calculate the nD view direction vector of the camera.

        Parameters
        ----------
        ndim : int
            Number of dimensions in which to embed the 3D view vector.
        dims_displayed : Tuple[int]
            Dimensions in which to embed the 3D view vector.

        Returns
        -------
        view_direction_nd : np.ndarray
            nD view direction vector as an (ndim, ) ndarray
        """
        if len(dims_displayed) != 3:
            return None
        view_direction_nd = np.zeros(ndim)
        view_direction_nd[list(dims_displayed)] = self.view_direction
        return view_direction_nd

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

    def from_legacy_angles(
        self, angles: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Convert camera angles from vispy convention (legacy behaviour) to napari.

        Vispy (and previously napari) uses YZX ordering, but in napari we use ZYX.
        Rotations are extrinsic.
        """
        # see #8281 for why this is yzx. In short: longstanding vispy bug.
        rot = R.from_euler('yzx', angles, degrees=True)
        # rotate 90 degrees to get neutral position at 0, 0, 0
        rot = rot * R.from_euler('x', -90, degrees=True)
        angles = rot.as_euler('zyx', degrees=True)
        # flip angles where orientation is flipped relative to default, so the
        # resulting rotation is always right-handed (i.e: CCW when facing the plane)
        flipped = angles * np.where(
            self._vispy_flipped_axes(ndisplay=3), -1, 1
        )
        return cast(tuple[float, float, float], tuple(flipped))

    def to_legacy_angles(
        self, angles: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Convert camera angles to napari convention to vispy (legacy behaviour).

        Vispy (and previously napari) uses YZX ordering, but in napari we use ZYX.
        Rotations are extrinsic.
        """
        # flip angles where orientation is flipped relative to default, so the
        # resulting rotation is always right-handed (i.e: CCW when facing the plane)
        flipped_angles = angles * np.where(
            self._vispy_flipped_axes(ndisplay=3), -1, 1
        )
        # see #8281 for why this is yzx. In short: longstanding vispy bug.
        rot = R.from_euler('zyx', flipped_angles, degrees=True)
        # flip angles so handedness of rotation is always right
        rot = rot * R.from_euler('x', 90, degrees=True)
        return cast(
            tuple[float, float, float],
            tuple(rot.as_euler('yzx', degrees=True)),
        )

    def _vispy_flipped_axes(
        self, ndisplay: Literal[2, 3] = 2
    ) -> tuple[int, int, int]:
        # Note: the Vispy axis order is xyz, or horizontal, vertical, depth,
        # while the napari axis order is zyx / plane-row-column, or depth, vertical,
        # horizontal — i.e. it is exactly inverted. This switch happens when data
        # is passed from napari to Vispy, usually with a transposition. In the camera
        # models, this means that the order of these orientations appear in the
        # opposite order to that in napari.components.Camera.
        #
        # Note that the default Vispy camera orientations come from Vispy, not from us.
        vispy_default_orientation = (
            ('right', 'up', 'towards')
            if ndisplay == 2
            else ('right', 'down', 'away')
        )

        # Vispy uses xyz coordinates; napari uses zyx coordinates. We therefore
        # start by inverting the order of coordinates coming from the napari
        # camera model:
        orientation_xyz = self.orientation[::-1]
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
