"""Camera inertia for smooth panning and rotation animation."""

from __future__ import annotations

from enum import Enum, auto
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QTimer

from napari._pydantic_compat import Field
from napari.utils.events.evented_model import EventedModel

if TYPE_CHECKING:
    import numpy.typing as npt

    from napari.components import Camera, Dims


class InertiaState(Enum):
    """State of the inertia system."""

    IDLE = auto()  # Not tracking or animating
    TRACKING = auto()  # Tracking mouse movement during drag
    ANIMATING = auto()  # Playing animation after release


class InertiaConfig(EventedModel):
    """Configuration for camera inertia behavior.

    All speed/velocity parameters are in canvas/screen space (pixels) to ensure
    consistent behavior regardless of data size or zoom level.

    Attributes
    ----------
    pan_friction : float
        Pan decay rate per second. Higher values cause faster deceleration.
        Default: 5.0
    pan_damping : float
        Fraction of velocity to apply for panning motion (0-1).
        Lower values reduce initial velocity. Default: 0.6
    pan_max_speed : float
        Maximum pan velocity in canvas pixels/second. Default: 200.0
    pan_min_speed : float
        Minimum pan speed (canvas pixels/sec) to trigger inertia animation. Default: 4.0
    pan_stop_speed : float
        Pan speed threshold (canvas pixels/sec) below which animation stops. Default: 2.5
    rotate_friction : float
        Rotation decay rate per second. Higher values cause faster deceleration.
        Default: 7.0 (slightly faster than pan for less dizziness)
    rotate_damping : float
        Fraction of velocity to apply for rotation (0-1).
        Rotation is usually more sensitive than panning. Default: 0.4
    rotate_max_speed : float
        Maximum rotation velocity in degrees per second. Default: 120.0
    rotate_min_speed : float
        Minimum rotation speed to trigger rotation inertia. Default: 1.5
    rotate_stop_speed : float
        Rotation speed threshold below which animation stops. Default: 1.0
    max_dt : float
        Maximum time (seconds) between last movement and release
        to trigger inertia. Prevents stale velocity from being used.
        Default: 0.1
    timer_interval_ms : int
        Interval in milliseconds for animation timer (~60 FPS). Default: 16
    """

    pan_friction: float = Field(
        4.5,
        ge=0.0,
        description='Pan decay rate per second. Higher values cause faster deceleration.',
    )
    pan_damping: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description='Fraction of velocity to apply for panning motion (0-1). Lower values reduce initial velocity.',
    )
    pan_max_speed: float = Field(
        800.0,
        gt=0.0,
        description='Maximum pan velocity in canvas pixels/second.',
    )
    pan_min_speed: float = Field(
        5.0,
        ge=0.0,
        description='Minimum pan speed (canvas pixels/sec) to trigger inertia animation.',
    )
    pan_stop_speed: float = Field(
        2.5,
        ge=0.0,
        description='Pan speed threshold (canvas pixels/sec) below which animation stops.',
    )
    rotate_friction: float = Field(
        6.0,
        ge=0.0,
        description='Rotation decay rate per second. Higher values cause faster deceleration.',
    )
    rotate_damping: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description='Fraction of velocity to apply for rotation (0-1). Rotation is usually more sensitive than panning.',
    )
    rotate_max_speed: float = Field(
        180.0,
        gt=0.0,
        description='Maximum rotation velocity in degrees per second.',
    )
    rotate_min_speed: float = Field(
        3.0,
        ge=0.0,
        description='Minimum rotation speed to trigger rotation inertia.',
    )
    rotate_stop_speed: float = Field(
        3.0,
        ge=0.0,
        description='Rotation speed threshold below which animation stops.',
    )
    max_dt: float = Field(
        0.06,
        gt=0.0,
        description='Maximum time (seconds) between last movement and release to trigger inertia.',
    )
    timer_interval_ms: int = Field(
        16,
        gt=0,
        description='Interval in milliseconds for animation timer (~60 FPS).',
    )


class CameraInertia:
    """Manages camera inertia for smooth panning and rotation animation.

    This class tracks mouse movement during drag operations and applies
    a smooth deceleration animation when the mouse is released, creating
    an inertial "coast" effect similar to mobile touch interfaces.

    Parameters
    ----------
    camera : Camera
        The napari camera model.
    dims : Dims
        The napari dims model.
    config : InertiaConfig, optional
        Configuration for inertia behavior. If None, uses defaults.

    Attributes
    ----------
    enabled : bool
        Whether inertia is enabled. Can be toggled at runtime.
    """

    def __init__(
        self, camera: Camera, dims: Dims, config: InertiaConfig | None = None
    ) -> None:
        self._camera = camera
        self._dims = dims
        self._config = config or InertiaConfig()
        self._state = InertiaState.IDLE

        # Tracking state
        self._last_pos: npt.NDArray[np.floating] | None = None
        self._last_angles: npt.NDArray[np.floating] | None = None
        self._last_time: float | None = None

        # Animation state
        self._pan_velocity: npt.NDArray[np.floating] | None = None
        self._rotate_velocity: npt.NDArray[np.floating] | None = None

        # Animation timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._animation_step)

        # Public control
        self.enabled = True

    @property
    def state(self) -> InertiaState:
        """Current state of the inertia system."""
        return self._state

    @property
    def is_animating(self) -> bool:
        """Whether inertia animation is currently playing."""
        return self._state == InertiaState.ANIMATING

    def on_press(self) -> None:
        """Handle mouse press event.

        Stops any ongoing animation and resets tracking state.
        """
        if not self.enabled:
            return

        self.stop()
        self._state = InertiaState.IDLE

    def on_drag(self) -> None:
        """Handle mouse drag event.

        Tracks position and angles for velocity calculation.
        Should be called during mouse movement while dragging.
        """
        if not self.enabled:
            return

        # Store position in canvas/screen space (zoom-independent)
        # to ensure consistent feel regardless of data size or zoom level
        pos = np.array(self._camera.center, dtype=np.float64) * self._camera.zoom
        now = perf_counter()

        self._last_pos = pos
        self._last_time = now

        # Only track rotation in 3D mode
        if self._dims.ndisplay == 3:
            angles = np.array(self._camera.angles, dtype=np.float64)
            self._last_angles = angles
        else:
            self._last_angles = None

        self._state = InertiaState.TRACKING

    def on_release(self) -> None:
        """Handle mouse release event.

        Calculates velocity from tracked movement and starts inertia
        animation if velocity is significant enough.
        """
        if not self.enabled or self._state != InertiaState.TRACKING:
            self._reset_tracking()
            return

        if self._last_pos is None or self._last_time is None:
            self._reset_tracking()
            return

        # Current position in canvas/screen space (zoom-independent)
        current_pos = np.array(self._camera.center, dtype=np.float64) * self._camera.zoom
        current_time = perf_counter()
        dt = current_time - self._last_time
        
        # Only start inertia if the release is recent enough
        if not (0.001 < dt < self._config.max_dt):
            self._reset_tracking()
            return

        # Calculate and apply pan velocity (in canvas space)
        pan_velocity = self._calculate_pan_velocity(current_pos, dt)

        # Calculate rotation velocity only in 3D mode
        rotate_velocity = None
        if self._dims.ndisplay == 3:
            current_angles = np.array(self._camera.angles, dtype=np.float64)
            rotate_velocity = self._calculate_rotate_velocity(
                current_angles, dt
            )

        # Start animation if either velocity is significant
        if pan_velocity is not None or rotate_velocity is not None:
            self._pan_velocity = pan_velocity
            self._rotate_velocity = rotate_velocity
            self._last_time = current_time
            self._state = InertiaState.ANIMATING
            self._timer.start(self._config.timer_interval_ms)
        else:
            self._reset_tracking()

    def stop(self) -> None:
        """Stop inertia animation and reset state."""
        if self._timer.isActive():
            self._timer.stop()
        self._pan_velocity = None
        self._rotate_velocity = None
        self._reset_tracking()
        if self._state == InertiaState.ANIMATING:
            self._state = InertiaState.IDLE

    def _reset_tracking(self) -> None:
        """Reset tracking state variables."""
        self._last_pos = None
        self._last_angles = None
        self._last_time = None
        if self._state == InertiaState.TRACKING:
            self._state = InertiaState.IDLE

    def _calculate_pan_velocity(
        self, current_pos: npt.NDArray[np.floating], dt: float
    ) -> npt.NDArray[np.floating] | None:
        """Calculate pan velocity for camera panning.

        Velocity is calculated in canvas/screen space (pixels/sec) to ensure
        consistent feel regardless of data size or zoom level.

        Parameters
        ----------
        current_pos : np.ndarray
            Current camera center position in canvas space (pos * zoom).
        dt : float
            Time delta since last tracked position.

        Returns
        -------
        np.ndarray or None
            Pan velocity vector in canvas space, or None if below threshold.
        """
        if self._last_pos is None:
            return None

        # Velocity in canvas/screen space (pixels per second)
        velocity = (current_pos - self._last_pos) / dt

        # Apply damping
        velocity = velocity * self._config.pan_damping

        # Cap maximum speed
        speed = np.linalg.norm(velocity)
        if speed > self._config.pan_max_speed:
            velocity = velocity * (self._config.pan_max_speed / speed)
            speed = self._config.pan_max_speed

        # Check if velocity is significant enough
        if speed < self._config.pan_min_speed:
            return None

        return velocity

    def _calculate_rotate_velocity(
        self, current_angles: npt.NDArray[np.floating], dt: float
    ) -> npt.NDArray[np.floating] | None:
        """Calculate rotation velocity for camera rotation.

        Parameters
        ----------
        current_angles : np.ndarray
            Current camera angles.
        dt : float
            Time delta since last tracked angles.

        Returns
        -------
        np.ndarray or None
            Rotation velocity vector (degrees/sec), or None if below threshold.
        """
        if self._last_angles is None:
            return None

        # Calculate angle difference, handling wrapping (shortest path)
        # This prevents issues when angles cross 0°/360° boundary
        angle_diff = current_angles - self._last_angles
        # Normalize to [-180, 180] range for each angle component
        angle_diff = (angle_diff + 180) % 360 - 180

        rotate_velocity = angle_diff / dt

        # Apply rotation-specific damping
        rotate_velocity = rotate_velocity * self._config.rotate_damping

        # Cap maximum rotation speed
        rotate_speed = np.linalg.norm(rotate_velocity)
        if rotate_speed > self._config.rotate_max_speed:
            rotate_velocity = rotate_velocity * (
                self._config.rotate_max_speed / rotate_speed
            )
            rotate_speed = self._config.rotate_max_speed

        # Check if rotation velocity is significant enough
        if rotate_speed < self._config.rotate_min_speed:
            return None

        return rotate_velocity

    def _animation_step(self) -> None:
        """Execute one step of inertia animation with friction."""
        # Verify we still have active velocities
        if self._pan_velocity is None and self._rotate_velocity is None:
            self.stop()
            return

        now = perf_counter()
        dt = now - self._last_time if self._last_time is not None else 0.0
        self._last_time = now

        # Apply pan velocity with pan-specific friction
        if self._pan_velocity is not None:
            pan_decay = np.exp(-self._config.pan_friction * dt)
            self._pan_velocity = self._pan_velocity * pan_decay

            # Convert canvas-space velocity to world-space displacement
            # displacement is in canvas pixels, convert to world units
            displacement_world = self._pan_velocity * dt / self._camera.zoom

            center = np.array(self._camera.center, dtype=np.float64)
            center = center + displacement_world
            self._camera.center = tuple(center)

            # Stop pan velocity if too small
            pan_speed = np.linalg.norm(self._pan_velocity)
            if pan_speed < self._config.pan_stop_speed:
                self._pan_velocity = None

        # Apply rotation velocity with rotation-specific friction (3D only)
        if self._rotate_velocity is not None and self._dims.ndisplay == 3:
            rotate_decay = np.exp(-self._config.rotate_friction * dt)
            self._rotate_velocity = self._rotate_velocity * rotate_decay
            angular_displacement = self._rotate_velocity * dt
            angles = np.array(self._camera.angles, dtype=np.float64)
            angles = angles + angular_displacement
            self._camera.angles = tuple(angles)

            # Stop rotation velocity if too small
            rotate_speed = np.linalg.norm(self._rotate_velocity)
            if rotate_speed < self._config.rotate_stop_speed:
                self._rotate_velocity = None

        # Stop timer if both velocities are done
        if self._pan_velocity is None and self._rotate_velocity is None:
            self.stop()
